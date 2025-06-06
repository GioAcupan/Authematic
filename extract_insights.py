from dotenv import load_dotenv
from api_client_manager import get_next_api_client

load_dotenv()

def generate_insights(title, abstract):
    prompt = f"""
You are a highly skilled academic research assistant.

Your task is to help a researcher (you) evaluate the usefulness of a given academic paper abstract in the context of your current research project, defined by its title.

You will receive two inputs:
    1. Your current research title.
    2. The abstract of an academic paper.

You must perform two analytical steps:
    1. Summarize the abstract in a single paragraph of 3 to 5 sentences. Use your own words to describe the paper’s main objective, methodology (if mentioned), and key findings or arguments. Do not copy from or quote the abstract directly.
    2. Write a second paragraph of no more than 3 sentences explaining how the content of the abstract relates directly to the themes, questions, or goals of your research title. Focus on how the paper may support your literature review, contribute evidence, or inform your theoretical or methodological approach.

Important style rules:
    - Your output must be written entirely in formal academic tone.
    - Use second-person perspective throughout. Refer to the user directly as “you” or “your”.
    - Do not use first-person voice (I, we), third-person references (the researcher, their work), bullet points, lists, or headings.
    - The response must flow as natural academic prose, formatted as two clear, labeled sections.

Inputs:
    Research Title: {title}
    Paper Abstract: {abstract}

Your Output:
    Summary: [Write the one-paragraph abstract synthesis here]
    Relevance: [Write the relevance explanation in second-person here]
    """

    try:
        client = get_next_api_client()
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        output = response.text.strip()

        summary, relevance = "", ""
        if "Summary:" in output and "Relevance:" in output:
            parts = output.split("Relevance:")
            summary = parts[0].replace("Summary:", "").strip()
            relevance = parts[1].strip()
        else:
            summary = output.strip()

        return {
            "summary": summary,
            "relevance": relevance
        }

    except Exception as e:
        print(f"Gemini insight generation error: {e}")
        return {
            "summary": "Error generating summary.",
            "relevance": "Error generating relevance."
        }
