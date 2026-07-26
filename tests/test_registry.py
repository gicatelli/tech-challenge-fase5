"""Testes para src/models/registry.py — Model Registry e governança."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.models.registry import (
    REQUIRED_TAGS,
    get_data_version,
    get_git_sha,
    get_production_model,
    promote_to_production,
    register_and_promote,
    register_champion_model,
)


class TestRequiredTags:
    """Testes para as tags obrigatórias de governança."""

    def test_has_model_name(self):
        """Tags devem incluir model_name."""
        assert "model_name" in REQUIRED_TAGS

    def test_has_model_version(self):
        """Tags devem incluir model_version."""
        assert "model_version" in REQUIRED_TAGS

    def test_has_owner(self):
        """Tags devem incluir owner."""
        assert "owner" in REQUIRED_TAGS

    def test_has_risk_level(self):
        """Tags devem incluir risk_level."""
        assert "risk_level" in REQUIRED_TAGS

    def test_has_training_data_version(self):
        """Tags devem incluir training_data_version."""
        assert "training_data_version" in REQUIRED_TAGS

    def test_minimum_required_tags(self):
        """Deve ter pelo menos 5 tags obrigatórias."""
        assert len(REQUIRED_TAGS) >= 5


class TestGetGitSha:
    """Testes para get_git_sha."""

    @patch("src.models.registry.subprocess.run")
    def test_returns_short_sha(self, mock_run):
        """Deve retornar SHA curto do commit."""
        mock_run.return_value = MagicMock(stdout="abc1234\n", returncode=0)
        sha = get_git_sha()
        assert sha == "abc1234"

    @patch("src.models.registry.subprocess.run")
    def test_returns_unknown_on_failure(self, mock_run):
        """Deve retornar 'unknown' quando git não disponível."""
        from subprocess import CalledProcessError

        mock_run.side_effect = CalledProcessError(1, "git")
        sha = get_git_sha()
        assert sha == "unknown"

    @patch("src.models.registry.subprocess.run")
    def test_returns_unknown_on_file_not_found(self, mock_run):
        """Deve retornar 'unknown' quando git não instalado."""
        mock_run.side_effect = FileNotFoundError()
        sha = get_git_sha()
        assert sha == "unknown"


class TestGetDataVersion:
    """Testes para get_data_version."""

    def test_returns_hash_for_existing_file(self, tmp_path):
        """Deve retornar hash MD5 curto para arquivo existente."""
        test_file = tmp_path / "data.csv"
        test_file.write_text("col1,col2\n1,2\n3,4\n")

        version = get_data_version(str(test_file))

        assert len(version) == 8
        assert version.isalnum()

    def test_returns_unknown_for_missing_file(self):
        """Deve retornar 'unknown' para arquivo inexistente."""
        version = get_data_version("/nonexistent/path/data.csv")
        assert version == "unknown"

    def test_same_content_same_hash(self, tmp_path):
        """Mesmo conteúdo deve gerar mesmo hash."""
        file1 = tmp_path / "file1.csv"
        file2 = tmp_path / "file2.csv"
        content = "a,b,c\n1,2,3\n"
        file1.write_text(content)
        file2.write_text(content)

        assert get_data_version(str(file1)) == get_data_version(str(file2))

    def test_different_content_different_hash(self, tmp_path):
        """Conteúdos diferentes devem gerar hashes diferentes."""
        file1 = tmp_path / "file1.csv"
        file2 = tmp_path / "file2.csv"
        file1.write_text("data1")
        file2.write_text("data2")

        assert get_data_version(str(file1)) != get_data_version(str(file2))


class TestRegisterChampionModel:
    """Testes para register_champion_model."""

    @patch("src.models.registry.MlflowClient")
    @patch("src.models.registry.mlflow")
    def test_registers_model_with_tags(self, mock_mlflow, mock_client_cls):
        """Deve registrar modelo com tags obrigatórias."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_model_details = MagicMock()
        mock_model_details.version = "3"
        mock_mlflow.register_model.return_value = mock_model_details

        with patch("src.models.registry.get_data_version", return_value="abc12345"):
            with patch("src.models.registry.get_git_sha", return_value="def6789"):
                version = register_champion_model(
                    run_id="test-run-123",
                    model_name="test-model",
                    model_version="2.0.0",
                    model_type="regression",
                    risk_level="medium",
                )

        assert version == "3"
        mock_mlflow.register_model.assert_called_once()

        # Verificar que set_model_version_tag foi chamado múltiplas vezes
        assert mock_client.set_model_version_tag.call_count >= 10

    @patch("src.models.registry.MlflowClient")
    @patch("src.models.registry.mlflow")
    def test_includes_timestamp(self, mock_mlflow, mock_client_cls):
        """Tags devem incluir timestamp de registro."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_model_details = MagicMock()
        mock_model_details.version = "1"
        mock_mlflow.register_model.return_value = mock_model_details

        with patch("src.models.registry.get_data_version", return_value="hash"):
            with patch("src.models.registry.get_git_sha", return_value="sha"):
                register_champion_model(run_id="run-1")

        # Verificar que registered_at foi setado
        tag_calls = mock_client.set_model_version_tag.call_args_list
        tag_keys = [call[1]["key"] if "key" in call[1] else call[0][2] for call in tag_calls]
        assert "registered_at" in tag_keys


class TestPromoteToProduction:
    """Testes para promote_to_production."""

    @patch("src.models.registry.MlflowClient")
    @patch("src.models.registry.mlflow")
    def test_transitions_to_production(self, mock_mlflow, mock_client_cls):
        """Deve transicionar modelo para Production."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        promote_to_production("test-model", version="2")

        mock_client.transition_model_version_stage.assert_called_once_with(
            name="test-model",
            version="2",
            stage="Production",
            archive_existing_versions=True,
        )

    @patch("src.models.registry.MlflowClient")
    @patch("src.models.registry.mlflow")
    def test_adds_promotion_tag(self, mock_mlflow, mock_client_cls):
        """Deve adicionar tag de promoção."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        promote_to_production("test-model", version="1")

        # Verificar que promoted_to_production_at foi setado
        mock_client.set_model_version_tag.assert_called()
        tag_call = mock_client.set_model_version_tag.call_args
        assert tag_call[1]["key"] == "promoted_to_production_at"

    @patch("src.models.registry.MlflowClient")
    @patch("src.models.registry.mlflow")
    def test_updates_description_when_provided(self, mock_mlflow, mock_client_cls):
        """Deve atualizar descrição quando fornecida."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        promote_to_production("model", version="1", description="Champion v1")

        mock_client.update_model_version.assert_called_once_with(
            name="model",
            version="1",
            description="Champion v1",
        )

    @patch("src.models.registry.MlflowClient")
    @patch("src.models.registry.mlflow")
    def test_no_description_no_update(self, mock_mlflow, mock_client_cls):
        """Não deve atualizar descrição quando vazia."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        promote_to_production("model", version="1", description="")

        mock_client.update_model_version.assert_not_called()


class TestRegisterAndPromote:
    """Testes para register_and_promote."""

    @patch("src.models.registry.promote_to_production")
    @patch("src.models.registry.register_champion_model")
    def test_calls_both_functions(self, mock_register, mock_promote):
        """Deve chamar registro e promoção."""
        mock_register.return_value = "5"

        version = register_and_promote(
            run_id="run-123",
            model_name="lstm-petr4",
            model_version="1.0.0",
        )

        assert version == "5"
        mock_register.assert_called_once_with(
            run_id="run-123",
            model_name="lstm-petr4",
            model_version="1.0.0",
        )
        mock_promote.assert_called_once()

    @patch("src.models.registry.promote_to_production")
    @patch("src.models.registry.register_champion_model")
    def test_passes_version_to_promote(self, mock_register, mock_promote):
        """Deve passar a versão correta para promoção."""
        mock_register.return_value = "7"

        register_and_promote(run_id="run-abc", model_name="model")

        promote_call_kwargs = mock_promote.call_args[1]
        assert promote_call_kwargs["version"] == "7"
        assert promote_call_kwargs["model_name"] == "model"


class TestGetProductionModel:
    """Testes para get_production_model."""

    @patch("src.models.registry.mlflow")
    def test_loads_model_from_registry(self, mock_mlflow):
        """Deve carregar modelo do registry com URI correto."""
        mock_model = MagicMock()
        mock_mlflow.pytorch.load_model.return_value = mock_model

        result = get_production_model("lstm-petr4-predictor")

        assert result == mock_model
        mock_mlflow.pytorch.load_model.assert_called_once_with(
            "models:/lstm-petr4-predictor/Production"
        )
