"""P8 GPU session: the only changed training semantic is a 3 O1 + 1 CMB batch."""

from __future__ import annotations

from typing import Any, Mapping

from src.opd.production_b2_formal_gpu_v2 import FormalB2SessionV2
from src.opd.production_qualification_two_step_gpu_v7 import ProductionTwoStepQualificationV6Error


class P8FormalB2Session(FormalB2SessionV2):
    """Exact P5.1 formula/runtime with the ADR-0039 source-mix override."""

    def _validate_source_roles_for_backward(
        self, source_roles: tuple[str, ...]
    ) -> None:
        """Replace only P4.8d's old 2:2 source-shape assertion for P8."""

        if sorted(source_roles) != [
            "medical_opd_cmb",
            "medical_opd_o1",
            "medical_opd_o1",
            "medical_opd_o1",
        ]:
            raise ProductionTwoStepQualificationV6Error(
                "P8 backward source batch differs from three O1 plus one CMB"
            )

    def _source_prompt_rows(self, prompt_rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        from src.opd.calibration_data import contains_forbidden_supervision

        if len(prompt_rows) != 4:
            raise ProductionTwoStepQualificationV6Error("P8 B2 requires exactly four source-real prompts")
        counts = {
            source: sum(row.get("target_role") == source for row in prompt_rows)
            for source in ("medical_opd_o1", "medical_opd_cmb")
        }
        if counts != {"medical_opd_o1": 3, "medical_opd_cmb": 1}:
            raise ProductionTwoStepQualificationV6Error("P8 B2 prompt batch is not 3 O1 plus 1 CMB")
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in prompt_rows:
            source_role = row.get("target_role")
            if (
                contains_forbidden_supervision(row)
                or not isinstance(source_role, str)
                or any(marker in source_role for marker in ("final", "controller", "confirmation", "label"))
            ):
                raise ProductionTwoStepQualificationV6Error("P8 prompt row contains forbidden supervision")
            sample_id = row.get("sample_id")
            content_hash = row.get("content_hash")
            if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
                raise ProductionTwoStepQualificationV6Error("P8 prompt identity is absent or duplicated")
            if not (
                isinstance(content_hash, str)
                and len(content_hash) == 64
                and all(character in "0123456789abcdef" for character in content_hash)
            ):
                raise ProductionTwoStepQualificationV6Error("P8 prompt content hash is invalid")
            seen.add(sample_id)
            prompt = self.render_prompt_text(row)
            result.append(
                {
                    "fixture_id": sample_id,
                    "source_role": source_role,
                    "source_sample_id": sample_id,
                    "content_hash": content_hash,
                    "prompt_ids": [
                        int(value)
                        for value in self.tokenizer.apply_chat_template(
                            [
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": prompt},
                            ],
                            tokenize=True,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                    ],
                }
            )
        return result


__all__ = ["P8FormalB2Session"]
