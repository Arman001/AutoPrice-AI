from fastapi import FastAPI
from api.routes import router


app = FastAPI(title="AutoPrice AI API")
app.include_router(router)


@app.get("/")
def home():
    return {"message": "Welcome to AutoPrice AI API"}
  

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)