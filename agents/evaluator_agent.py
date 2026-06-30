from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from state import State, EvaluatorOutput

MAX_ITERATIONS = 3

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
evaluator_llm = llm.with_structured_output(EvaluatorOutput)


def evaluator_agent(state: State) -> dict:
    """
    Évalue la réponse du RAG et décide si elle est approuvée.
    Force l'approbation si le nombre max d'itérations est atteint.
    """
    iteration = state.get("iteration", 0) + 1

    # Protection anti-boucle infinie
    if iteration >= MAX_ITERATIONS:
        return {
            "feedback": "Nombre maximum d'itérations atteint. Envoi forcé.",
            "approved": True,
            "iteration": iteration,
        }

    system = SystemMessage(content="""Tu es un évaluateur rigoureux.
Analyse la réponse fournie et décide si elle est satisfaisante.
Critères d'approbation :
- La réponse est directement liée à la question
- Elle est basée sur le contexte fourni, pas sur des hallucinations
- Elle est claire et complète
""")

    user = HumanMessage(content=f"""Question originale : {state['query']}

Documents récupérés :
{chr(10).join(state['retrieved_docs'])}

Réponse générée :
{state['rag_response']}

Évalue cette réponse.
""")

    result: EvaluatorOutput = evaluator_llm.invoke([system, user])

    return {
        "feedback": result.feedback,
        "approved": result.approved,
        "iteration": iteration,
    }