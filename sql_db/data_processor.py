"""
Process JSON fields from TMDB CSV files.
"""
import json
import ast
from typing import List, Dict, Any, Optional
from logging_config.logger import get_logger

logger = get_logger(__name__)


class DataProcessor:
    """Process and extract data from JSON-like strings in CSV."""

    @staticmethod
    def parse_json_field(field_value: Any) -> List[Dict[str, Any]]:
        """
        Parse JSON field from CSV (handles both JSON and string representation).

        Args:
            field_value: Field value from CSV

        Returns:
            Parsed list of dictionaries
        """
        if not field_value or field_value == '[]':
            return []

        try:
            # Try JSON parsing first
            return json.loads(field_value)
        except (json.JSONDecodeError, TypeError):
            try:
                # Try ast.literal_eval (safer than eval)
                return ast.literal_eval(field_value)
            except (ValueError, SyntaxError):
                logger.warning(f"Failed to parse field: {str(field_value)[:100]}")
                return []

    @staticmethod
    def extract_names(json_list: List[Dict[str, Any]]) -> List[str]:
        """
        Extract 'name' field from list of dicts.

        Args:
            json_list: List of dictionaries with 'name' field

        Returns:
            List of names
        """
        names = []
        for item in json_list:
            if isinstance(item, dict) and 'name' in item:
                names.append(item['name'])
        return names

    @staticmethod
    def extract_director(crew_json: str) -> Optional[str]:
        """
        Extract director name from crew JSON.

        Args:
            crew_json: JSON string of crew members

        Returns:
            Director name or None

        Example:
            crew_json = '[{"job": "Director", "name": "James Cameron"}, ...]'
            Returns: "James Cameron"
        """
        crew_list = DataProcessor.parse_json_field(crew_json)

        for crew_member in crew_list:
            if isinstance(crew_member, dict):
                if crew_member.get('job') == 'Director':
                    return crew_member.get('name')

        return None

    @staticmethod
    def extract_cast_names(cast_json: str, limit: Optional[int] = None) -> List[str]:
        """
        Extract cast member names from cast JSON.

        Args:
            cast_json: JSON string of cast members
            limit: Maximum number of cast members to extract

        Returns:
            List of cast member names

        Example:
            cast_json = '[{"name": "Sam Worthington", "order": 0}, ...]'
            Returns: ["Sam Worthington", "Zoe Saldana", ...]
        """
        cast_list = DataProcessor.parse_json_field(cast_json)
        cast_names = DataProcessor.extract_names(cast_list)

        if limit:
            return cast_names[:limit]
        return cast_names

    @staticmethod
    def extract_crew_names(crew_json: str) -> List[str]:
        """
        Extract all crew member names from crew JSON.

        Args:
            crew_json: JSON string of crew members

        Returns:
            List of crew member names
        """
        crew_list = DataProcessor.parse_json_field(crew_json)
        return DataProcessor.extract_names(crew_list)

    @staticmethod
    def extract_genres(genres_json: str) -> List[str]:
        """
        Extract genre names.

        Args:
            genres_json: JSON string of genres

        Returns:
            List of genre names
        """
        genres_list = DataProcessor.parse_json_field(genres_json)
        return DataProcessor.extract_names(genres_list)

    @staticmethod
    def extract_keywords(keywords_json: str) -> List[str]:
        """
        Extract keyword names.

        Args:
            keywords_json: JSON string of keywords

        Returns:
            List of keywords
        """
        keywords_list = DataProcessor.parse_json_field(keywords_json)
        return DataProcessor.extract_names(keywords_list)

    @staticmethod
    def extract_production_companies(prod_companies_json: str) -> List[str]:
        """
        Extract production company names.

        Args:
            prod_companies_json: JSON string of production companies

        Returns:
            List of company names
        """
        companies_list = DataProcessor.parse_json_field(prod_companies_json)
        return DataProcessor.extract_names(companies_list)

    @staticmethod
    def extract_year_from_date(release_date: str) -> Optional[int]:
        """
        Extract year from release date string.

        Args:
            release_date: Date string (e.g., "2009-12-10")

        Returns:
            Year as integer or None
        """
        if not release_date or release_date == '':
            return None

        try:
            # Assuming format YYYY-MM-DD
            year_str = release_date.split('-')[0]
            return int(year_str)
        except (ValueError, IndexError):
            logger.debug(f"Could not extract year from: {release_date}")
            return None
