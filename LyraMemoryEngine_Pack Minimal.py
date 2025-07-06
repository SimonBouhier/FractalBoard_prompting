# LyraMemoryEngine — Pack Minimal « Sémantique Riche »
# --------------------------------------------------------
# ↑ Version intégrale : extraction de syntagmes, co‑présence,
#   projection Concret/Abstrait & Naturel/Artificiel, CRITRIX multipolaire,
#   et génération de miroirs narratifs.
# --------------------------------------------------------

import re, random, math
from collections import defaultdict, Counter

# ──────────────────────────────────────────────────────────
# 1.  UTILITAIRES LÉGERS
# ──────────────────────────────────────────────────────────
STOPWORDS = {
    "dans","le","la","les","des","de","d","du","un","une","que","qui","qu","en","au","aux","et","a","à","l","pour","plus","avec","par","sur","nos","notre","leur","leurs"
}

AXE_CONCRET_ABSTRAIT = {
    "outil":-0.8,"marteau":-0.9,"table":-0.85,"chaise":-0.85,"forêt":-0.6,
    "justice":+0.9,"liberté":+0.9,"mémoire":+0.4,"conscience":+0.8,
}

AXE_NATUREL_ARTIFICIEL = {
    "forêt":-0.9,"arbre":-0.8,"rivière":-0.9,"écosystème":-0.8,
    "ville":+0.8,"béton":+0.9,"robot":+0.95,"algorithme":+0.8,
}

# ──────────────────────────────────────────────────────────
# 2.  CRITRIX 2.0 — DISPERSION + CO‑PRÉSENCE + FACETTES
# ──────────────────────────────────────────────────────────
class CritrixMirroir:
    def __init__(self, coherence_thr=0.6, ambiguity_thr=0.4, dispersion_thr=0.12):
        self.c_thr = coherence_thr
        self.a_thr = ambiguity_thr
        self.d_thr = dispersion_thr

    def valence(self, κ, ambiguity, echo_grad, social_w):
        return κ*(1-ambiguity) - echo_grad - 0.2*social_w

    @staticmethod
    def dispersion(vec, all_vec):
        if not all_vec: return 0
        return sum(math.dist(vec, v) for v in all_vec)/len(all_vec)

    def juger(self, concept, κ, ambiguity, echo_grad, social_w, vec, all_vec):
        v = self.valence(κ, ambiguity, echo_grad, social_w)
        disp = self.dispersion(vec, all_vec)
        if disp>self.d_thr:
            return f"🔮 {concept} possède une dispersion riche ({disp:.2f}). Plusieurs facettes latentes. Veux‑tu en explorer une ?"
        if v<self.c_thr:
            if ambiguity>self.a_thr:
                return f"🪞 {concept} paraît dilué. Valence {v:.2f}. Faut‑il le densifier ou le disséquer ?"
            return f"💡 {concept} gagnerait à être précisé. Valence {v:.2f}. Reformuler ?"
        return None

# ──────────────────────────────────────────────────────────
# 3.  LYRA ENGINE AVEC PACK MINIMAL
# ──────────────────────────────────────────────────────────
class LyraMemoryEngine:
    def __init__(self):
        self.memory = []
        self.co_presence = defaultdict(int)
        self.critrix = CritrixMirroir()

    def embed(self, concept:str):
        base = [random.random() for _ in range(108)]
        perceptual = 1 if concept in AXE_CONCRET_ABSTRAIT else 0.5
        return base+[perceptual]

    def extraire_syntagmes(self, texte:str):
        mots = re.findall(r"\b\w+\b", texte.lower())
        mots = [m for m in mots if m not in STOPWORDS and len(m)>3]
        bigrams = [f"{mots[i]} {mots[i+1]}" for i in range(len(mots)-1)]
        raw = mots+bigrams
        return [w for w,_ in Counter(raw).most_common(40)]

    def axes_projection(self, concept:str):
        return AXE_CONCRET_ABSTRAIT.get(concept,0), AXE_NATUREL_ARTIFICIEL.get(concept,0)

    def injecter_texte(self, texte:str):
        concepts = self.extraire_syntagmes(texte)
        for c in concepts:
            κ = round(random.uniform(0.6,0.9),2)
            amb = round(random.uniform(0.1,0.4),2)
            echo = round(random.uniform(0.05,0.2),2)
            vec = self.embed(c)
            self.memory.append({"concept":c,"vec":vec,"κ":κ,"amb":amb,"echo":echo,"axes":self.axes_projection(c)})
            for o in concepts:
                if o!=c: self.co_presence[tuple(sorted((c,o)))] +=1
            soc = sum(self.co_presence[tuple(sorted((c,o)))] for o in concepts if o!=c)/len(concepts)
            msg = self.critrix.juger(c,κ,amb,echo,soc,vec,[m['vec'] for m in self.memory])
            if msg: print("[CRITRIX]", msg)

    def carte_axes(self):
        print("\nAxes Concret(−)/Abstrait(+) × Naturel(−)/Artificiel(+):")
        for m in self.memory:
            c,a = m['axes']
            print(f"  {m['concept']:<25} CA:{c:+.2f}  NA:{a:+.2f}")

# ──────────────────────────────────────────────────────────
# 4.  DEMO RAPIDE EXÉCUTABLE
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo = LyraMemoryEngine()
    demo.injecter_texte(
        "Dans le silence des forêts primaires, l'homme, porteur d'un esprit ardent, a commencé à façonner le monde. "
        "Chaque pierre taillée, chaque outil forgé, chaque bâtiment érigé portait l'empreinte de sa volonté. "
        "Le temps tissait son œuvre, mêlant la nature luxuriante à l'artifice croissant des cités. "
        "De cette interaction constante, une mémoire collective s'est ancrée, une conscience s'est éveillée, "
        "scrutant les ombres et les lumières d'une réalité toujours plus complexe. "
        "Aujourd'hui, alors que nos algorithmes explorent des dimensions insoupçonnées, l'écho de cette genèse résonne encore. "
        "La quête de liberté et de justice, ces abstractions fondamentales, continue de guider nos pas, "
        "même lorsque les machines s'interrogent sur le sens profond des silences que nous laissons derrière nous."
    )
    demo.carte_axes()
