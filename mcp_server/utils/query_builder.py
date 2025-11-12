"""
SQL query builder utility.
"""
from typing import Optional


class QueryBuilder:
    """Helper for building complex SQL queries."""

    @staticmethod
    def build_where_clause(
        genre: Optional[str] = None,
        year: Optional[int] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        cast: Optional[str] = None,
        director: Optional[str] = None,
        title: Optional[str] = None
    ) -> tuple:
        """
        Build SQL WHERE clause with parameters.

        Returns:
            (query_fragment, parameters)
        """
        conditions = []
        params = []

        if genre:
            conditions.append("genre LIKE ?")
            params.append(f"%{genre}%")

        if year:
            conditions.append("year = ?")
            params.append(year)

        if year_min:
            conditions.append("year >= ?")
            params.append(year_min)

        if year_max:
            conditions.append("year <= ?")
            params.append(year_max)

        if cast:
            conditions.append("cast LIKE ?")
            params.append(f"%{cast}%")

        if director:
            conditions.append("director LIKE ?")
            params.append(f"%{director}%")

        if title:
            conditions.append("title LIKE ?")
            params.append(f"%{title}%")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        return where_clause, params
