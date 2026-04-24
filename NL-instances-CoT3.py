"""
NL-instances-CoT3  —  SPARQL-based CoT generation (PCA + SHACL)
================================================================
Identical to NL-instances-CoT2 but additionally reads a SHACL
validation report and prepends a [VALID] / [INVALID] tag to each
instance based on whether the head entity passes SHACL constraints.

Extra config key  (kg_sparql block)
------------------------------------
  shacl_report     path to SHACL validation report .ttl file

Usage
-----
  python NL-instances-CoT3v5.py --config config.json
"""

import argparse
import json
import math
import os
import re
from pathlib import Path

import pandas as pd
from rdflib import Graph, Namespace, URIRef


# ---------------------------------------------------------------------------
# Helpers  (shared with CoT2)
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def is_valid(value) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, str) and value.strip() == '':
        return False
    return True


def safe_str(value) -> str:
    return str(value).strip() if is_valid(value) else ''


def format_predicate(predicate: str) -> str:
    if not predicate:
        return ''
    natural = re.sub(r'([A-Z])', r' \1', predicate).strip().lower()
    if natural.startswith('has '):
        natural = natural[4:]
    return natural


def format_entity(entity: str) -> str:
    return entity.replace('_', ' ') if entity else ''


def local_name(uri_str: str) -> str:
    s = str(uri_str)
    if '#' in s:
        return s.split('#')[-1]
    if '/' in s:
        return s.split('/')[-1]
    return s


# ---------------------------------------------------------------------------
# Rule parsing
# ---------------------------------------------------------------------------

def parse_body(body_str: str):
    if not is_valid(body_str):
        return []
    return [(m.group(1), m.group(2), m.group(3))
            for m in re.finditer(r'(\?\w+)\s+(\S+)\s+(\S+)', body_str)]


def parse_head(head_str: str):
    if not is_valid(head_str):
        return None, None, None
    parts = head_str.strip().split()
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    return None, None, None


# ---------------------------------------------------------------------------
# SPARQL query generation
# ---------------------------------------------------------------------------

def build_sparql_query(head_parsed, body_parsed,
                       ns_prefix: str, namespace: str) -> str | None:
    head_var, head_pred, head_obj = head_parsed
    if not head_pred or not body_parsed:
        return None

    all_vars = set()
    for bv, _, bo in body_parsed:
        all_vars.add(bv)
        if bo.startswith('?'):
            all_vars.add(bo)
    all_vars.add(head_var)
    if head_obj.startswith('?'):
        all_vars.add(head_obj)

    select_vars = ' '.join(sorted(all_vars))

    def term(t: str) -> str:
        return t if t.startswith('?') else f'{ns_prefix}:{t}'

    body_lines = '\n'.join(
        f'    {term(bv)} {ns_prefix}:{bp} {term(bo)} .'
        for bv, bp, bo in body_parsed
    )
    head_line = f'    {term(head_var)} {ns_prefix}:{head_pred} {term(head_obj)} .'

    return (
        f'PREFIX {ns_prefix}: <{namespace}>\n'
        f'SELECT DISTINCT {select_vars} ?_head_exists WHERE {{\n'
        f'{body_lines}\n'
        f'    OPTIONAL {{\n'
        f'{head_line}\n'
        f'        BIND(true AS ?_head_exists)\n'
        f'    }}\n'
        f'}}'
    )


# ---------------------------------------------------------------------------
# SHACL violation index
# ---------------------------------------------------------------------------

def load_shacl_violations(report_path: str) -> set:
    """
    Extract violated entity local-names from a SHACL validation report.
    Entities NOT in the returned set are considered [VALID].

    Uses direct regex extraction on the raw file text — consistent with
    the previous script's approach and immune to non-standard TTL syntax
    (e.g. unbound empty prefix ':' produced by some SHACL validators).

    Matches both:
      sh:focusNode <http://example.org/Entity>
      sh:focusNode prefix:EntityName
    """
    if not report_path or not os.path.exists(report_path):
        print(f"  WARNING: SHACL report not found at {report_path}. "
              f"All entities will be tagged [VALID].")
        return set()

    print(f"Loading SHACL report from {report_path} ...")
    violated = set()
    try:
        with open(report_path, encoding='utf-8', errors='replace') as f:
            content = f.read()
        # Match sh:focusNode followed by either a full URI <...> or a prefixed name
        for m in re.finditer(r'sh:focusNode\s+(?:<([^>]+)>|([\w][\w.:/-]*))', content):
            uri_match   = m.group(1)   # full URI inside <>
            qname_match = m.group(2)   # prefixed name e.g. yago:Entity
            if uri_match:
                violated.add(local_name(uri_match))
            elif qname_match:
                violated.add(qname_match.split(':')[-1])
        print(f"  SHACL violations loaded: {len(violated)} entities with violations")
    except Exception as e:
        print(f"  ERROR reading SHACL report: {e}. "
              f"All entities will be tagged [VALID].")
    return violated


# ---------------------------------------------------------------------------
# Main converter class
# ---------------------------------------------------------------------------

class RuleToNaturalLanguageSPARQLCoT3:

    def __init__(self, rules_csv: str, kg_file: str, kg_name: str,
                 namespace: str, namespace_prefix: str,
                 shacl_report: str | None = None,
                 pca_threshold: float = 0.5):
        self.kg_name          = kg_name
        self.namespace        = namespace.rstrip('/') + '/'
        self.namespace_prefix = namespace_prefix
        self.pca_threshold    = pca_threshold

        # Load SHACL violations index
        self.violated_entities = load_shacl_violations(shacl_report or '')

        # Load and clean rules
        self.rules_df = pd.read_csv(rules_csv)
        self._clean_rules()
        print(f"Valid rules to process: {len(self.rules_df)}")

        # Load KG into RDFLib in-memory store
        print(f"Loading KG from {kg_file} ...")
        self.graph = Graph()
        fmt = 'nt' if str(kg_file).endswith('.nt') else 'turtle'
        self.graph.parse(str(kg_file), format=fmt)
        print(f"KG loaded: {len(self.graph)} triples")

        # Output directory
        self.output_dir = Path(f"CoT/CoT3_{kg_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── internal helpers ────────────────────────────────────────────────────

    def _clean_rules(self):
        original = len(self.rules_df)
        mask = (
            self.rules_df['Body'].notna() &
            self.rules_df['Head'].notna() &
            self.rules_df['Body'].astype(str).str.strip().str.startswith('?') &
            self.rules_df['Head'].astype(str).str.strip().str.startswith('?')
        )
        self.rules_df = self.rules_df[mask].reset_index(drop=True)
        removed = original - len(self.rules_df)
        if removed:
            print(f"Removed {removed} invalid rows from rules dataframe")

    def _get_pca(self, rule_index) -> float | None:
        for col in ['Pca_Confidence', 'PCA_Confidence',
                    'PCA Confidence', 'pca_confidence']:
            if col in self.rules_df.columns:
                val = self.rules_df.iloc[rule_index][col]
                if is_valid(val):
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        pass
        return None

    def _get_stat(self, rule_index, cols, fmt):
        for col in cols:
            if col in self.rules_df.columns:
                val = self.rules_df.iloc[rule_index][col]
                if is_valid(val):
                    return fmt(val)
        return None

    def _run_sparql(self, query_str: str):
        try:
            return list(self.graph.query(query_str))
        except Exception as e:
            print(f"    SPARQL error: {e}")
            return []

    def _validity_tag(self, entity_local_name: str) -> str:
        """Return [VALID] or [INVALID] based on SHACL violations."""
        return '[INVALID]' if entity_local_name in self.violated_entities else '[VALID]'

    # ── rule file generation ────────────────────────────────────────────────

    def generate_rule_file(self, rule_index: int) -> str | None:
        row  = self.rules_df.iloc[rule_index]
        head = safe_str(row['Head'])
        body = safe_str(row['Body'])

        head_parsed = parse_head(head)
        body_parsed = parse_body(body)
        head_var, head_pred, head_obj = head_parsed

        if not head_pred or not body_parsed:
            return None

        # PCA classification text
        pca = self._get_pca(rule_index)
        if pca is not None:
            cls      = "POSITIVE" if pca >= self.pca_threshold else "NEGATIVE"
            op       = ">=" if pca >= self.pca_threshold else "<"
            pca_text = (f"The path is classified as {cls} "
                        f"(PCA Confidence {pca:.4f} {op} "
                        f"threshold {self.pca_threshold})")
        else:
            cls      = "UNKNOWN"
            op       = "?"
            pca_text = "Classification unknown (PCA Confidence not available)"

        # Build and execute SPARQL query
        query_str = build_sparql_query(
            head_parsed, body_parsed,
            self.namespace_prefix, self.namespace
        )
        if query_str is None:
            return None

        rows = self._run_sparql(query_str)

        # NL rule header
        body_desc = " AND ".join(
            f"{format_predicate(bp)} {format_entity(bo)}"
            for _, bp, bo in body_parsed
        )
        head_desc = f"{format_predicate(head_pred)} {format_entity(head_obj)}"

        nl  = f"Rule {rule_index + 1}:\n"
        nl += f"If {body_desc}, then {head_desc}.\n\n"
        nl += f"Formal Rule:\nHead: {head}\nBody: {body}\n\n"
        nl += f"Real Instances from Knowledge Graph ({len(rows)} found):\n\n"

        if rows:
            for row_result in rows:
                # Use asdict() — ResultRow is a tuple, labels are integer-indexed
                # so row_result[string] raises KeyError. asdict() is the correct API.
                bindings = {}
                for vname, val in row_result.asdict().items():
                    if val is not None:
                        bindings[vname] = local_name(str(val))

                entity     = bindings.get(head_var.lstrip('?'), '')
                head_value = (bindings.get(head_obj.lstrip('?'), '')
                              if head_obj.startswith('?') else head_obj)
                answer     = 'yes' if bindings.get('_head_exists') == 'true' else 'no'

                # SHACL validity tag based on head entity
                validity_tag = self._validity_tag(entity)

                # Instance text:
                #   yes -> head fact first, then body facts
                #   no  -> entity name only, then body facts
                if answer == 'yes':
                    instance_text = (
                        f"{format_entity(entity)} has "
                        f"{format_predicate(head_pred)} "
                        f"{format_entity(head_value)}"
                    )
                else:
                    instance_text = format_entity(entity)

                # Body facts with correct subject per atom
                for bv, bp, bo in body_parsed:
                    subj_val = bindings.get(bv.lstrip('?'), '')
                    obj_val  = (bindings.get(bo.lstrip('?'), '')
                                if bo.startswith('?') else bo)
                    if subj_val and obj_val:
                        instance_text += (
                            f", {format_entity(subj_val)} has "
                            f"{format_predicate(bp)} "
                            f"{format_entity(obj_val)}"
                        )

                # CoT3 format: [VALID/INVALID] then PCA classification
                instance_text += f". {validity_tag} {pca_text} Answer: {answer}"
                nl += instance_text + "\n"
        else:
            nl += "No matching instances found in the Knowledge Graph.\n"

        # Rule statistics footer
        nl += "\nRule Statistics:\n"
        if pca is not None:
            nl += f"- PCA Confidence: {pca:.4f}\n"
            nl += (f"- Rule Classification: {cls} "
                   f"(PCA Confidence {pca:.4f} {op} "
                   f"threshold {self.pca_threshold})\n")
        else:
            nl += "- PCA Confidence: Not available\n"
            nl += "- Rule Classification: UNKNOWN\n"

        v = self._get_stat(rule_index,
                           ['Standard_Confidence', 'Std_Confidence', 'std_confidence'],
                           lambda x: f"{float(x):.4f}")
        if v:
            nl += f"- Standard Confidence: {v}\n"

        v = self._get_stat(rule_index,
                           ['Support', 'Positive_Examples', 'support'],
                           lambda x: str(int(x)))
        if v:
            nl += f"- Positive Examples: {v}\n"

        v = self._get_stat(rule_index,
                           ['Head Coverage', 'Head_Coverage', 'head_coverage'],
                           lambda x: f"{float(x):.4f}")
        if v:
            nl += f"- Head Coverage: {v}\n"

        return nl

    # ── batch conversion ────────────────────────────────────────────────────

    def convert_all_rules(self, max_rules: int | None = None):
        total = (len(self.rules_df) if max_rules is None
                 else min(max_rules, len(self.rules_df)))
        print(f"\nProcessing {total} rules ...")

        successful = failed = skipped = total_instances = 0

        for idx in range(total):
            row = self.rules_df.iloc[idx]
            if not is_valid(row['Head']) or not is_valid(row['Body']):
                print(f"  Skipping rule {idx+1}: invalid Head/Body")
                skipped += 1
                continue
            try:
                nl_text = self.generate_rule_file(idx)
                if nl_text:
                    fpath = self.output_dir / f"rule_{idx+1}.txt"
                    fpath.write_text(nl_text, encoding='utf-8')
                    successful += 1
                    m = re.search(r'\((\d+) found\)', nl_text)
                    if m:
                        total_instances += int(m.group(1))
                    if (idx + 1) % 50 == 0:
                        avg = total_instances / max(successful, 1)
                        print(f"  Processed {idx+1}/{total} rules "
                              f"(avg instances: {avg:.1f})")
                else:
                    failed += 1
            except Exception as e:
                import traceback
                print(f"  Error on rule {idx+1}: {e}")
                traceback.print_exc()
                failed += 1

        print(f"\nConversion complete!")
        print(f"  Successfully converted : {successful}")
        print(f"  Skipped (invalid)      : {skipped}")
        print(f"  Failed                 : {failed}")
        print(f"  Total instances found  : {total_instances}")
        print(f"  Output saved to        : {self.output_dir.absolute()}")

    def create_summary(self):
        pca_col = next(
            (c for c in ['Pca_Confidence', 'PCA_Confidence',
                         'PCA Confidence', 'pca_confidence']
             if c in self.rules_df.columns),
            None
        )
        summary_path = self.output_dir / "rules_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"{self.kg_name.upper()} CoT3 RULES — SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total Rules          : {len(self.rules_df)}\n")
            f.write(f"Output Directory     : {self.output_dir.absolute()}\n")
            f.write(f"KG Triples           : {len(self.graph)}\n")
            f.write(f"PCA Threshold        : {self.pca_threshold}\n")
            f.write(f"Namespace            : {self.namespace}\n")
            f.write(f"Namespace Prefix     : {self.namespace_prefix}\n")
            f.write(f"SHACL violations idx : {len(self.violated_entities)} entities\n\n")
            if pca_col:
                valid_pca = self.rules_df[pca_col].dropna()
                pos = (valid_pca >= self.pca_threshold).sum()
                neg = (valid_pca < self.pca_threshold).sum()
                nan = self.rules_df[pca_col].isna().sum()
                f.write("PCA Confidence Statistics:\n")
                f.write(f"  Positive (>= {self.pca_threshold}) : {pos}\n")
                f.write(f"  Negative (<  {self.pca_threshold}) : {neg}\n")
                if nan:
                    f.write(f"  Missing PCA              : {nan}\n")
        print(f"Summary saved to: {summary_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NL-instances CoT3 generator (SPARQL-based, PCA + SHACL)"
    )
    parser.add_argument("--config", required=True, help="Path to config.json")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    sparql = cfg.get("kg_sparql", {})

    kg_file          = sparql.get("kg_file")
    rules_csv        = sparql.get("rules_csv")
    namespace        = sparql.get("namespace")
    namespace_prefix = sparql.get("namespace_prefix", "ex")
    shacl_report     = sparql.get("shacl_report", "")
    pca_threshold    = cfg["data_generation"].get("pca_threshold", 0.5)
    kg_name          = Path(cfg["data_dir"]).name  # e.g. "YAGO3-10"

    for label, val in [("kg_file", kg_file),
                        ("rules_csv", rules_csv),
                        ("namespace", namespace)]:
        if not val:
            raise ValueError(
                f"Missing required kg_sparql.{label} in config.json"
            )
    if not os.path.exists(kg_file):
        raise FileNotFoundError(f"KG file not found: {kg_file}")
    if not os.path.exists(rules_csv):
        raise FileNotFoundError(f"Rules CSV not found: {rules_csv}")

    print(f"\nNL-instances-CoT3  (SPARQL-based)")
    print(f"  KG file          : {kg_file}")
    print(f"  Rules CSV        : {rules_csv}")
    print(f"  Namespace        : {namespace}")
    print(f"  Namespace prefix : {namespace_prefix}")
    print(f"  SHACL report     : {shacl_report or '(none)'}")
    print(f"  PCA threshold    : {pca_threshold}\n")

    converter = RuleToNaturalLanguageSPARQLCoT3(
        rules_csv        = rules_csv,
        kg_file          = kg_file,
        kg_name          = kg_name,
        namespace        = namespace,
        namespace_prefix = namespace_prefix,
        shacl_report     = shacl_report,
        pca_threshold    = pca_threshold,
    )
    converter.convert_all_rules()
    converter.create_summary()
    print("\nDone! Check the output folder for individual rule files.")
