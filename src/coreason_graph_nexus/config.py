# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_graph_nexus

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings managed by Pydantic Settings.
    Reads from environment variables and optional .env file.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Neo4j Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="URI for Neo4j database")
    neo4j_user: str = Field(default="neo4j", description="Username for Neo4j")
    neo4j_password: str = Field(default="password", description="Password for Neo4j")
    neo4j_database: str = Field(default="neo4j", description="Target database name")

    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379/0", description="URL for Redis cache")

    # App Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    default_batch_size: int = Field(default=10000, description="Default batch size for bulk operations")


settings = Settings()
