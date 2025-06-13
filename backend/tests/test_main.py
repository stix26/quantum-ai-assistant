from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()

@app.get('/')
async def root():
    return {'message': 'ok'}

client = TestClient(app)

def test_root_endpoint():
    resp = client.get('/')
    assert resp.status_code == 200
    assert resp.json() == {'message': 'ok'}
