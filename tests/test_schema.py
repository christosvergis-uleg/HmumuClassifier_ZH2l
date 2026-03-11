from eventclf.config import HMUMU_ZH2L_SCHEMA

def test_schema_basic():
    assert HMUMU_ZH2L_SCHEMA.label
    assert len(HMUMU_ZH2L_SCHEMA.feature_names()) > 0
    assert len(set(HMUMU_ZH2L_SCHEMA.feature_names())) == len(HMUMU_ZH2L_SCHEMA.feature_names())

def test_schema_has_unique_features():
    names = HMUMU_ZH2L_SCHEMA.feature_names()
    assert len(names) > 0
    assert len(set(names)) == len(names), "Duplicate feature names in schema"