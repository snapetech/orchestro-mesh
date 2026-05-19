from orchestro_mesh.redaction import redact_text


def test_redacts_generic_token():
    result = redact_text("api_key=supersecret123")
    assert "REDACTED" in result.text
    assert "generic_token" in result.findings
