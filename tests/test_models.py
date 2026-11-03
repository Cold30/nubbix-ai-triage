
import pytest
from pydantic import ValidationError
from src.models import TriageResponse


def test_respuesta_valida():
    response = TriageResponse(
        answer="Revisaremos tu factura.",
        confidence=0.95,
        category="billing",
        actions=["Revisar cobro"]
    )

    assert response.category == "billing"
    assert response.confidence == 0.95


def test_confianza_invalida():
    with pytest.raises(ValidationError):
        TriageResponse(
            answer="Revisaremos tu factura.",
            confidence=1.75,
            category="billing",
            actions=[]
        )


def test_categoria_invalida():
    with pytest.raises(ValidationError):
        TriageResponse(
            answer="Revisaremos tu factura.",
            confidence=0.8,
            category="pokemon",
            actions=[]
        )
