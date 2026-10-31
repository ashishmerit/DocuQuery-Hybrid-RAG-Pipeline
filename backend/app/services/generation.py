import json

from app.models import (
    QueryResponse,
    RetrievalResult,
    SourceCitation,
)
from app.services.document_store import document_store
from app.services.llm_client import LLMClient, LLMProviderError
from app.services.prompt_builder import (
    SYSTEM_PROMPT,
    build_user_prompt,
)


INSUFFICIENT_ANSWER = (
    "The uploaded document does not contain enough information "
    "to answer this question."
)


class AnswerGenerationService:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    async def generate(
        self,
        question: str,
        results: list[RetrievalResult],
    ) -> QueryResponse:

        if not results:
            return QueryResponse(
                status="insufficient_evidence",
                question=question,
                answer=INSUFFICIENT_ANSWER,
                citations=[],
            )

        user_prompt, source_map = build_user_prompt(
            question,
            results,
        )

        raw_response = await self.llm_client.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        try:
            payload = json.loads(raw_response)

            answer = payload["answer"]
            citation_ids = payload["citation_ids"]
            insufficient = payload["insufficient_evidence"]

            if (
                not isinstance(answer, str)
                or not isinstance(citation_ids, list)
                or not isinstance(insufficient, bool)
                or not all(
                    isinstance(item, str)
                    for item in citation_ids
                )
            ):
                raise ValueError("Invalid response fields.")

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise LLMProviderError(
                "The LLM returned malformed structured answer data."
            ) from exc

        if insufficient or not answer.strip():
            return QueryResponse(
                status="insufficient_evidence",
                question=question,
                answer=answer.strip() or INSUFFICIENT_ANSWER,
                citations=[],
            )

        # Resolve only source IDs that were actually supplied to the model.
        valid_ids = list(dict.fromkeys(
            source_id
            for source_id in citation_ids
            if source_id in source_map
        ))

        if not valid_ids:
            return QueryResponse(
                status="insufficient_evidence",
                question=question,
                answer=INSUFFICIENT_ANSWER,
                citations=[],
            )

        citations = []

        for source_id in valid_ids:
            result = source_map[source_id]
            chunk = result.chunk

            document = document_store.get(chunk.document_id)
            document_name = (
                document.filename if document else "Unknown document"
            )

            citations.append(
                SourceCitation(
                    citation_id=source_id,
                    document_id=chunk.document_id,
                    document_name=document_name,
                    chunk_id=chunk.chunk_id,
                    page_number=chunk.page_number,
                    excerpt=chunk.text[:600],
                )
            )

        return QueryResponse(
            status="answered",
            question=question,
            answer=answer.strip(),
            citations=citations,
        )