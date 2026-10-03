from pathlib import Path

from pdf_rag_agent.config import AppConfig


def test_config_builds_paths_from_project_root(
    tmp_path: Path,
) -> None:
    config = AppConfig.from_project_root(tmp_path)

    assert config.project_root == tmp_path.resolve()
    assert config.data_directory == tmp_path.resolve() / "data"
    assert config.chroma_directory == tmp_path.resolve() / "chroma_db"
    assert config.collection_name == "pdf_documents"


def test_config_creates_required_directories(
    tmp_path: Path,
) -> None:
    config = AppConfig.from_project_root(tmp_path)

    config.create_directories()

    assert config.data_directory.is_dir()
    assert config.chroma_directory.is_dir()