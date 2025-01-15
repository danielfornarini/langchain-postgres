"""Copied over from langchain_community.

This code should be moved to langchain proper or removed entirely.
"""
import json
import logging
from typing import List, Union

import numpy as np
from sqlalchemy import JSON
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import JSONB

logger = logging.getLogger(__name__)

Matrix = Union[List[List[float]], List[np.ndarray], np.ndarray]


def cosine_similarity(X: Matrix, Y: Matrix) -> np.ndarray:
    """Row-wise cosine similarity between two equal-width matrices."""
    if len(X) == 0 or len(Y) == 0:
        return np.array([])

    X = np.array(X)
    Y = np.array(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Number of columns in X and Y must be the same. X has shape {X.shape} "
            f"and Y has shape {Y.shape}."
        )
    try:
        import simsimd as simd  # type: ignore

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        Z = 1 - np.array(simd.cdist(X, Y, metric="cosine"))
        return Z
    except ImportError:
        logger.debug(
            "Unable to import simsimd, defaulting to NumPy implementation. If you want "
            "to use simsimd please install with `pip install simsimd`."
        )
        X_norm = np.linalg.norm(X, axis=1)
        Y_norm = np.linalg.norm(Y, axis=1)
        # Ignore divide by zero errors run time warnings as those are handled below.
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
        similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
        return similarity


def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: list,
    lambda_mult: float = 0.5,
    k: int = 4,
) -> List[int]:
    """Calculate maximal marginal relevance."""
    if min(k, len(embedding_list)) <= 0:
        return []
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    similarity_to_query = cosine_similarity(query_embedding, embedding_list)[0]
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    selected = np.array([embedding_list[most_similar]])
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        similarity_to_selected = cosine_similarity(embedding_list, selected)
        for i, query_score in enumerate(similarity_to_query):
            if i in idxs:
                continue
            redundant_score = max(similarity_to_selected[i])
            equation_score = (
                lambda_mult * query_score - (1 - lambda_mult) * redundant_score
            )
            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i
        idxs.append(idx_to_add)
        selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)
    return idxs


class PostgresCompiler(postgresql.dialect().statement_compiler):
    """Custom compiler class to handle JSONB and other complex data types"""

    def render_literal_value(self, value, type_):
        """
        Custom literal value renderer that handles JSONB and other PostgreSQL-specific types.
        """
        # Handle JSONB and JSON types
        if isinstance(type_, (JSONB, JSON)):
            return f"'{json.dumps(value)}'::jsonb"

        # For all other types, use the default rendering
        return super().render_literal_value(value, type_)


def get_postgres_sql(statement):
    """
    Convert a SQLAlchemy statement to a PostgreSQL string.
    Handles complex types like JSONB.

    Args:
        statement: A SQLAlchemy statement object

    Returns:
        str: The equivalent PostgreSQL query string

    Example:
        from sqlalchemy import select, Column, String, JSONB
        from sqlalchemy.ext.declarative import declarative_base

        Base = declarative_base()

        class Document(Base):
            __tablename__ = 'documents'
            id = Column(String, primary_key=True)
            data = Column(JSONB)

        # Create a sample query with JSONB
        query = select(Document).where(Document.data.contains({'value': 'spreadsheet'}))

        # Convert to PostgreSQL string
        sql_string = get_postgres_sql(query)
    """
    # Create a PostgreSQL dialect instance
    dialect = postgresql.dialect()

    # Compile the statement using our custom compiler
    compiled = statement.compile(
        dialect=dialect,
        compile_kwargs={
            "literal_binds": True,
            "statement_compiler": PostgresCompiler
        }
    )

    return str(compiled)