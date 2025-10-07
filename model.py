import google.generativeai as genai

genai.configure(api_key="AIzaSyDpjbjud9SxNK3UzAg6IX4AF6UDTdinpUA")

for model in genai.list_models():
    if "generateContent" in model.supported_generation_methods:
        print(model.name)
