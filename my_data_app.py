import json
import os
import re
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#from streamlit_annotated_text import annotated_text

from wordcloud import WordCloud
from datetime import datetime
import imaplib
import email
import anthropic
import base64
from email.header import decode_header

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Classification d'E-mails Clients",
    page_icon="✉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === 1. Initialisation du modèle ===
@st.cache_resource
def load_model():
    """Initialise le client Claude"""
    try:
        client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        return client
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation de Claude: {str(e)}")
        return None

# === 2. Gestion des données ===
def load_dataset():
    """Charge les données depuis le fichier JSONL local ou utilise des exemples par défaut."""
    try:
        data = []
        with open('data/emails_clients_annotes.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    except FileNotFoundError:
        st.warning("Fichier de base non trouvé, utilisation des exemples par défaut.")
        return [
            {"email": "Mon produit ne fonctionne pas...", "label": "support"},
            {"email": "Problème de connexion...", "label": "technique"},
            {"email": "Facture incorrecte du mois dernier", "label": "facturation"},
            {"email": "Comment annuler mon abonnement?", "label": "support"},
            {"email": "Erreur 404 sur la page de paiement", "label": "technique"},
            {"email": "Demande de remboursement", "label": "facturation"}
        
        ]
    except Exception as e:
        st.error(f"Erreur de lecture du fichier : {str(e)}")
        return None
    
    
def save_to_jsonl(new_data):
    """Ajoute de nouvelles entrées au fichier JSONL."""
    try:
        with open('data/emails_clients_annotes.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(new_data, ensure_ascii=False) + '\n')
        return True
    except Exception as e:
        st.error(f"Erreur lors de la sauvegarde : {str(e)}")
        return False

# === 3. Classificateur ===
class EmailClassifier:
    def __init__(self, client):
        self.client = client
        self.valid_labels = {"support", "facturation", "technique", "autres"}
        self.label_colors = {
            "support": "#FF6B6B",
            "facturation": "#4ECDC4",
            "technique": "#45B7D1",
            "autres": "#A5A5A5"
        }
        self.label_descriptions = {
            "support": "Questions sur l'utilisation, demandes d'aide",
            "facturation": "Problèmes de paiement, factures, remboursements",
            "technique": "Bugs, erreurs, problèmes techniques",
            "autres": "Non classé ou hors sujet"
        }

    def build_prompt(self, examples, email_to_classify):
        """Construit le prompt pour Claude"""
        prompt = """Tu es un expert en classification d'e-mails. Classe cet e-mail en une des catégories suivantes:
    - support: questions sur l'utilisation, demandes d'aide
    - facturation: problèmes de paiement, factures, remboursements
    - technique: bugs, erreurs, problèmes techniques
    - autres: pour tout ce qui ne correspond pas aux catégories ci-dessus

    Réponds UNIQUEMENT par le nom de la catégorie (support, facturation, technique ou autres), sans autre texte.

    Exemples:\n\n"""
        
        for ex in examples:
            prompt += f"E-mail: \"{ex['email']}\"\nCatégorie: {ex['label']}\n\n"
        
        prompt += f"E-mail: \"{email_to_classify}\"\nCatégorie:"
        return prompt

    def predict(self, email_text, examples, num_generations=3):
        """Prédit la catégorie d'un e-mail avec Claude"""
        prompt = self.build_prompt(examples, email_text)
        
        try:
            predictions = []
            for _ in range(num_generations):
                response = self.client.messages.create(
                    model="claude-3-haiku-20240307",  # ou "claude-3-haiku-20240307" pour un modèle plus léger
                    max_tokens=100,
                    temperature=0.3,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                # Extraction de la réponse
                response_text = response.content[0].text
                match = re.search(r"(support|facturation|technique|autres)", response_text.lower())
                if match:
                    label = match.group(1).lower()
                    if label not in self.valid_labels:
                        label = "autres"
                    predictions.append(label)

            if not predictions:
                return "autres", 0.0

            most_common, count = Counter(predictions).most_common(1)[0]
            confidence = count / num_generations
            return most_common, confidence
            
        except Exception as e:
            st.error(f"Erreur lors de la prédiction avec Claude: {str(e)}")
            return "autres", 0.0

    

    def evaluate(self, test_set, examples):
        """Évalue les performances du modèle"""
        y_true, y_pred, confidences = [], [], []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, sample in enumerate(test_set):
            status_text.text(f"Traitement de l'e-mail {i+1}/{len(test_set)}...")
            progress_bar.progress((i + 1) / len(test_set))
            
            pred, confidence = self.predict(sample["email"], examples)
            y_true.append(sample["label"])
            y_pred.append(pred)
            confidences.append(confidence)
            
            time.sleep(0.1)  # Pour éviter de surcharger l'API

        status_text.text("Évaluation terminée!")
        progress_bar.empty()
        
        # Calcul des métriques
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred, labels=list(self.valid_labels))
        
        return {
            "accuracy": accuracy,
            "report": report,
            "confusion_matrix": cm,
            "confidences": confidences
        }

# === 4. Fonctions de visualisation ===
def plot_confusion_matrix(cm, class_names):
    """Affiche la matrice de confusion"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title("Matrice de Confusion")
    return fig

def plot_metrics(report):
    """Affiche les métriques de classification"""
    metrics = ["precision", "recall", "f1-score"]
    classes = [k for k in report.keys() if k not in ["accuracy", "macro avg", "weighted avg"]]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, metric in enumerate(metrics):
        values = [report[cls][metric] for cls in classes]
        axes[i].bar(classes, values)
        axes[i].set_title(metric.capitalize())
        axes[i].set_ylim(0, 1.1)
        axes[i].grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    return fig

def generate_wordcloud(data, label):
    """Génère un nuage de mots pour une catégorie"""
    texts = [item["email"] for item in data if item["label"] == label]
    if not texts:
        return None
        
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color="white",
        colormap="viridis"
    ).generate(" ".join(texts))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Mots fréquents - {label.capitalize()}")
    return fig

# === 5. Fonctionnalités Gmail ===
def fetch_emails(email_address, password, limit=10):
    """Récupère les e-mails depuis Gmail"""
    try:
        # Connexion au serveur IMAP
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(email_address, password)
        mail.select("inbox")

        # Recherche des derniers e-mails
        status, messages = mail.search(None, "ALL")
        if status != "OK":
            st.error("Erreur lors de la récupération des e-mails")
            return []

        email_ids = messages[0].split()[-limit:]
        emails = []
        
        for e_id in email_ids:
            _, msg_data = mail.fetch(e_id, "(RFC822)")
            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)
            
            # Décodage du sujet
            subject, encoding = decode_header(msg["Subject"])[0] if msg["Subject"] else ("", None)
            if isinstance(subject, bytes):
                subject = subject.decode(encoding if encoding else "utf-8")
            
            # Extraction du texte
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type == "text/plain":
                        charset = part.get_content_charset() or 'utf-8'
                        body = part.get_payload(decode=True).decode(charset, errors='replace')
                        break
            else:
                charset = msg.get_content_charset() or 'utf-8'
                body = msg.get_payload(decode=True).decode(charset, errors='replace')
            
            emails.append({
                "id": e_id.decode(),
                "subject": subject,
                "body": body,
                "from": msg["From"],
                "date": msg["Date"]
            })
        
        mail.close()
        mail.logout()
        return emails
    
    except Exception as e:
        st.error(f"Erreur lors de la récupération des e-mails: {str(e)}")
        return []

# === 6. Interface Streamlit ===
def main():
    # Initialisation de l'état de session
    if "dataset" not in st.session_state:
        st.session_state.dataset = load_dataset()
    
    if "classifier" not in st.session_state:
        client = load_model()
        if client:
            st.session_state.classifier = EmailClassifier(client)
        else:
            st.error("Impossible de charger Claude. Vérifiez votre clé API Anthropic.")
            st.stop()
    
    # Sidebar
    with st.sidebar:
        st.title("Paramètres")
        
        # Upload de données
        # st.subheader("Données d'entraînement")
        # uploaded_file = st.file_uploader(
        #     "Télécharger un fichier JSONL", 
        #     type=["jsonl"],
        #     help="Fichier JSONL avec des e-mails annotés (champs 'email' et 'label')"
        # )
        # 
        # if uploaded_file:
        #     st.session_state.dataset = load_dataset(uploaded_file)
        #     st.success(f"{len(st.session_state.dataset)} e-mails chargés!")

        st.session_state.dataset = load_dataset()
        
        # Paramètres de classification
        st.subheader("Paramètres de classification")
        num_generations = st.slider(
            "Nombre de générations", 
            min_value=1, 
            max_value=5, 
            value=3,
            help="Nombre de prédictions à faire pour chaque e-mail (majorité gagnante)"
        )
        
        # Connexion Gmail
        st.subheader("Connexion Gmail")
        gmail_user = st.text_input("Adresse Gmail")
        gmail_pass = st.text_input("Mot de passe", type="password")
        fetch_btn = st.button("Récupérer les e-mails")
        
        # Aide
        st.markdown("---")
        st.info("""
        **Guide d'utilisation:**
        1. Chargez des e-mails annotés ou utilisez les exemples par défaut
        2. Testez la classification avec des e-mails manuels
        3. Évaluez les performances ou traitez des e-mails Gmail
        """)
    
    # Main content
    st.title("✉️ Classification d'E-mails Clients")
    st.markdown("""
    Classifiez automatiquement les e-mails clients en catégories (support, facturation, technique) 
    en utilisant le modèle Claude.
    """)
    
    # Onglets
    tab1, tab2, tab3, tab4 = st.tabs([
        "Classification", 
        "Évaluation", 
        "Exploration", 
        "Gmail"
    ])
    
    # Onglet 1: Classification
    with tab1:
        st.subheader("Tester la classification")
        
        col1, col2 = st.columns(2)
        
        with col1:
            email_text = st.text_area(
                "Entrez un e-mail à classifier",
                height=200,
                placeholder="Bonjour, j'ai un problème avec ma facture du mois dernier..."
            )
            
            if st.button("Classer"):
                if email_text.strip():
                    with st.spinner("Analyse en cours..."):
                        label, confidence = st.session_state.classifier.predict(
                            email_text, 
                            st.session_state.dataset,
                            num_generations
                        )
                        
                        st.success(f"Catégorie prédite: **{label}** (confiance: {confidence:.2f})")
                        
                        # Affichage annoté
                        st.markdown(
                                f'<div style="background-color: {st.session_state.classifier.label_colors[label]}; padding: 10px; border-radius: 5px;">'
                                f'<strong>{label}:</strong> {email_text}'
                                '</div>',
                                unsafe_allow_html=True
                            )
                        
                        
                        # Description de la catégorie
                        st.info(st.session_state.classifier.label_descriptions[label])
                else:
                    st.warning("Veuillez entrer un e-mail à classifier")
        
        with col2:
            st.subheader("Ajouter un exemple annoté")
            new_email = st.text_area("Nouvel e-mail", height=100)
            new_label = st.selectbox(
                "Catégorie", 
                options=list(st.session_state.classifier.label_descriptions.keys())
            )
            
            if st.button("Ajouter aux données d'entraînement"):
                if new_email.strip():
                    st.session_state.dataset.append({
                        "email": new_email,
                        "label": new_label
                    })
                    st.success("Exemple ajouté!")
                else:
                    st.warning("Veuillez entrer un e-mail")
    
    # Onglet 2: Évaluation
    with tab2:
        st.subheader("Évaluation des performances")
        
        if len(st.session_state.dataset) < 10:
            st.warning("Ajoutez au moins 10 e-mails annotés pour l'évaluation")
        else:
            train_size = st.slider(
                "Nombre d'exemples d'entraînement", 
                min_value=5, 
                max_value=len(st.session_state.dataset)-5, 
                value=min(30, len(st.session_state.dataset)-5)
            )
            
            if st.button("Lancer l'évaluation"):
                train_set = st.session_state.dataset[:train_size]
                test_set = st.session_state.dataset[train_size:]
                
                results = st.session_state.classifier.evaluate(test_set, train_set)
                
                # Affichage des résultats
                st.metric("Précision globale", f"{results['accuracy']:.2%}")
                
                # Métriques détaillées
                st.subheader("Rapport de classification")
                report_df = pd.DataFrame(results["report"]).transpose()
                st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)
                
                # Visualisations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.pyplot(plot_confusion_matrix(
                        results["confusion_matrix"],
                        list(st.session_state.classifier.valid_labels)
                    ))
                
                with col2:
                    st.pyplot(plot_metrics(results["report"]))
                
                # Téléchargement du rapport
                report_str = f"Rapport de classification\n\n"
                report_str += f"Précision globale: {results['accuracy']:.2%}\n\n"
                report_str += "Métriques par classe:\n"
                
                for cls in results["report"]:
                    if cls not in ["accuracy", "macro avg", "weighted avg"]:
                        report_str += f"\n{cls}:\n"
                        report_str += f"  Précision: {results['report'][cls]['precision']:.2f}\n"
                        report_str += f"  Rappel: {results['report'][cls]['recall']:.2f}\n"
                        report_str += f"  F1-score: {results['report'][cls]['f1-score']:.2f}\n"
                
                st.download_button(
                    "Télécharger le rapport complet",
                    data=report_str,
                    file_name=f"classification_report_{datetime.now().strftime('%Y%m%d')}.txt"
                )
    
    # Onglet 3: Exploration des données
    with tab3:
        st.subheader("Exploration des données annotées")
        
        if not st.session_state.dataset:
            st.warning("Aucune donnée disponible")
        else:
            # Statistiques de base
            df = pd.DataFrame(st.session_state.dataset)
            st.write(f"**Total d'e-mails:** {len(df)}")
            
            # Répartition des classes
            st.subheader("Répartition des catégories")
            fig, ax = plt.subplots(figsize=(8, 4))
            df["label"].value_counts().plot(kind="bar", color=[
                st.session_state.classifier.label_colors.get(x, "#333333") for x in df["label"].value_counts().index
            ], ax=ax)
            ax.set_title("Nombre d'e-mails par catégorie")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            
            # Nuages de mots par catégorie
            st.subheader("Mots caractéristiques par catégorie")
            selected_label = st.selectbox(
                "Sélectionnez une catégorie", 
                options=list(st.session_state.classifier.valid_labels)
            )
            
            wc_fig = generate_wordcloud(st.session_state.dataset, selected_label)
            if wc_fig:
                st.pyplot(wc_fig)
            else:
                st.warning(f"Aucun e-mail trouvé pour la catégorie {selected_label}")
            
            # Aperçu des données
            st.subheader("Aperçu des données")
            st.dataframe(df, use_container_width=True)
            
            # === Fonction pour convertir les données en format JSONL ===
            def convert_to_jsonl(data):
                """Convertit les données en mémoire en chaîne JSONL"""
                return "\n".join([json.dumps(item, ensure_ascii=False) for item in data])

            # === Dans l'interface (remplacez le bouton existant) ===
            st.download_button(
                "Télécharger les données annotées",
                data=convert_to_jsonl(st.session_state.dataset),  # Utilise la conversion directe
                file_name=f"data/emails_clients_annotes.jsonl",
                mime="text/plain",  # Plus approprié que application/json pour JSONL
                help="Téléchargez l'ensemble des e-mails classés au format JSONL"
            )
    # Onglet 4: Gmail
    with tab4:
        st.subheader("Classification des e-mails Gmail")
        
        if fetch_btn and gmail_user and gmail_pass:
            with st.spinner("Récupération des e-mails..."):
                emails = fetch_emails(gmail_user, gmail_pass)
                
                if emails:
                    st.success(f"{len(emails)} e-mails récupérés!")
                    
                    # Affichage et classification des e-mails
                    for i, email_msg in enumerate(emails):
                        with st.expander(f"E-mail {i+1}: {email_msg['subject']}"):
                            st.write(f"**De:** {email_msg['from']}")
                            st.write(f"**Date:** {email_msg['date']}")
                            
                            # Classification
                            full_text = f"{email_msg['subject']}\n{email_msg['body']}"
                            label, confidence = st.session_state.classifier.predict(
                                full_text, 
                                st.session_state.dataset,
                                num_generations
                            )
                            
                            # Affichage du résultat
                            st.write(f"**Catégorie prédite:** :{label}: **{label}** (confiance: {confidence:.2f})")
                            st.info(st.session_state.classifier.label_descriptions[label])
                            
                            # Affichage du contenu avec mise en évidence
                            st.text_area(
                                f"Contenu de l'e-mail {i+1}", 
                                value=full_text, 
                                height=200,
                                key=f"email_{i}"
                            )
                            
                            # Bouton pour ajouter aux données d'entraînement
                            if st.button(f"Annoter cet e-mail", key=f"annotate_{i}"):
                                actual_label = st.selectbox(
                                    "Corriger la catégorie si nécessaire",
                                    options=list(st.session_state.classifier.valid_labels),
                                    index=list(st.session_state.classifier.valid_labels).index(label),
                                    key=f"correction_{i}"
                                )
                                
                                if st.button(f"Confirmer l'annotation", key=f"confirm_{i}"):
                                    st.session_state.dataset.append({
                                        "email": full_text,
                                        "label": actual_label
                                    })
                                    st.success("E-mail ajouté aux données d'entraînement!")
                else:
                    st.warning("Aucun e-mail trouvé ou erreur de connexion")
        elif fetch_btn:
            st.warning("Veuillez entrer vos identifiants Gmail")

if __name__ == "__main__":
    main()