from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "DocuMind"
    OPENAI_API_KEY: str = ""
    DEBUG: bool = False
    
    # Paths
    UPLOAD_DIR: str = "data/uploads"
    INDEX_DIR: str = "data/index"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
