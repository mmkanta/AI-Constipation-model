# AI-Constipation-model
- install dependencies
```
pip install -r requirements.txt
```
- start application with reload (port 7000)
```
uvicorn main:app --reload --port 7000
```
- start application without reload (port 7000)
```
python main.py
```
```
uvicorn main:app --port 7000 --workers 2
```
- Swagger API
```
localhost:7000/docs
```
```
localhost:7000/redoc
```
- Docker (port 7000)
```
docker build . -t model
```
```
docker run -d --name model -p 7000:7000 model
```
