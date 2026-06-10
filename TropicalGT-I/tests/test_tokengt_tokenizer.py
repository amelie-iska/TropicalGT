from tropicalgt.records import GraphRecord
from tropicalgt.tokenizer import TokenGTTokenizer


def test_simple_graph_token_counts():
    graph = {"nodes": [{"id": "a"}, {"id": "b"}], "edges": [{"source": "a", "target": "b", "type": "rel"}]}
    batch = TokenGTTokenizer().batch_encode([GraphRecord("r", "hello", graph_json=graph)])
    assert batch.graph_token_counts.tolist() == [4]
    assert batch.node_counts.tolist() == [2]
    assert batch.edge_counts.tolist() == [1]
    assert batch.token_type_ids[0, :4].tolist() == [2, 0, 0, 1]
    assert batch.endpoint_ids[0, 3].tolist() == [0, 1]


def test_relabel_preserves_counts_and_types():
    g1 = {"nodes": [{"id": "a"}, {"id": "b"}], "edges": [{"source": "a", "target": "b"}]}
    g2 = {"nodes": [{"id": "x"}, {"id": "y"}], "edges": [{"source": "x", "target": "y"}]}
    tok = TokenGTTokenizer()
    b = tok.batch_encode([GraphRecord("r1", "a", graph_json=g1), GraphRecord("r2", "a", graph_json=g2)])
    assert b.graph_token_counts.tolist() == [4, 4]
    assert b.token_type_ids[0, :4].tolist() == b.token_type_ids[1, :4].tolist()
