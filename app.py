from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from scripts.hub_module import SherlockHub
from src.austen_slm.data_loader import prepare_data
import config.configs as configs


app = FastAPI()
templates = Jinja2Templates(directory="templates")


_, _, tokenizer = prepare_data(corpus_type='austen')
base_bot = SherlockHub(mode="base")
fine_bot = SherlockHub(mode="fine_tuned")
rag_bot = SherlockHub(mode="rag", repo_id=configs.REPO_ID)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate", response_class=HTMLResponse)
async def generate_results(request: Request, prompt: str = Form(...)):
    
    base_res = base_bot.generate(tokenizer=tokenizer, prompt=prompt)
    fine_res = fine_bot.generate(tokenizer=tokenizer, prompt=prompt)
    rag_res = rag_bot.generate(prompt=prompt)
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prompt": prompt,
        "base": base_res,
        "fine": fine_res,
        "rag": rag_res
    })
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)