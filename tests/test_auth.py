import pytest
from fastapi import HTTPException

from orchestro_mesh.auth import make_requester_dep, make_shared_token_dep, parse_bearer


def test_parse_bearer_extracts_token():
    assert parse_bearer("Bearer abc123") == "abc123"
    assert parse_bearer("bearer abc123") == "abc123"


def test_parse_bearer_rejects_garbage():
    assert parse_bearer(None) is None
    assert parse_bearer("") is None
    assert parse_bearer("Basic xyz") is None
    assert parse_bearer("Bearer ") is None


def test_requester_dep_with_api_tokens_maps_to_user():
    dep = make_requester_dep(mesh_token=None, api_tokens={"tok-keith": "keith"})
    assert dep(authorization="Bearer tok-keith", x_orchestro_requester=None) == "keith"


def test_requester_dep_with_api_tokens_rejects_unknown():
    dep = make_requester_dep(mesh_token=None, api_tokens={"tok-keith": "keith"})
    with pytest.raises(HTTPException) as exc:
        dep(authorization="Bearer wrong", x_orchestro_requester=None)
    assert exc.value.status_code == 401


def test_requester_dep_with_mesh_token_uses_header():
    dep = make_requester_dep(mesh_token="cluster", api_tokens={})
    assert dep(authorization="Bearer cluster", x_orchestro_requester="alice") == "alice"
    assert dep(authorization="Bearer cluster", x_orchestro_requester=None) == "anonymous"


def test_requester_dep_with_mesh_token_rejects_mismatch():
    dep = make_requester_dep(mesh_token="cluster", api_tokens={})
    with pytest.raises(HTTPException):
        dep(authorization="Bearer nope", x_orchestro_requester=None)


def test_requester_dep_open_when_unconfigured():
    dep = make_requester_dep(mesh_token=None, api_tokens={})
    assert dep(authorization=None, x_orchestro_requester="keith") == "keith"
    assert dep(authorization=None, x_orchestro_requester=None) == "anonymous"


def test_shared_token_dep_enforces_match():
    dep = make_shared_token_dep("secret")
    dep(authorization="Bearer secret")
    with pytest.raises(HTTPException):
        dep(authorization="Bearer wrong")
    with pytest.raises(HTTPException):
        dep(authorization=None)


def test_shared_token_dep_open_when_unset():
    dep = make_shared_token_dep(None)
    dep(authorization=None)
    dep(authorization="Bearer anything")
