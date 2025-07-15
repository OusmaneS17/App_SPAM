import json
import pickle
import streamlit as st
import imaplib
import email
from email.header import decode_header
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib 

# Configuration de la page
st.set_page_config(
    page_title="Détection de Spam",
    page_icon="✉️",
    layout="wide"
)

# === 1. Initialisation du modèle ===
@st.cache_resource
def load_model():
    """Charge le modèle de détection de spam et les composants nécessaires"""
    try:
        # Charger le modèle Naive Bayes
        model = joblib.load('./model/naive_bayes_model.joblib')

        # Charger les noms de fonctionnalités (si nécessaire)
        feature_names = joblib.load('./model/feature_names.joblib')

        # Charger le vectorizer (si vous l'avez sauvegardé séparément)
        vectorizer = joblib.load('./model/tfidf_vectorizer.joblib')
        
        # Créer un pipeline avec le vectorizer et le modèle
        pipeline = make_pipeline(vectorizer, model)
        
        return pipeline
    except Exception as e:
        # Retourner un modèle vide en cas d'échec
        return make_pipeline(TfidfVectorizer(), MultinomialNB())


# === Le reste de votre code reste inchangé ===
class SpamClassifier:
    def __init__(self, model):
        self.model = model
        self.labels = {"spam": "Message indésirable", "ham": "Message légitime"}
        self.colors = {"spam": "#FF6B6B", "ham": "#4ECDC4"}

    def train(self, emails, labels):
        """Entraîne le modèle et le sauvegarde"""
        self.model.fit(emails, labels)
        # Sauvegarder en .joblib si vous voulez garder la cohérence
        joblib.dump(self.model, './modele/naive_bayes_model.joblib')

    def predict(self, email_text):
        """Prédit si un email est spam ou non"""
        pred = self.model.predict([email_text])[0]
        proba = self.model.predict_proba([email_text])[0].max()
        return pred, proba
    


# === 3. Fonctionnalités Gmail ===
def fetch_emails(email_address, password, limit=10):
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(email_address, password)
        mail.select("inbox")

        _, messages = mail.search(None, "ALL")
        email_ids = messages[0].split()[-limit:]
        emails = []
        
        for e_id in email_ids:
            _, msg_data = mail.fetch(e_id, "(RFC822)")
            msg = email.message_from_bytes(msg_data[0][1])  # ← fonctionne maintenant correctement

            subject, encoding = decode_header(msg["Subject"])[0] if msg["Subject"] else ("", None)
            if isinstance(subject, bytes):
                subject = subject.decode(encoding or "utf-8")
            
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8', errors='replace')
                        break
            else:
                body = msg.get_payload(decode=True).decode(msg.get_content_charset() or 'utf-8', errors='replace')
            
            emails.append({
                "subject": subject,
                "body": body,
                "from": msg["From"],
                "date": msg["Date"]
            })
        
        mail.close()
        mail.logout()
        return emails
    except Exception as e:
        st.error(f"Erreur: {str(e)}")
        return []

# === 4. Interface ===
def main():
    # Initialisation
    model = load_model()
    classifier = SpamClassifier(model)
    
    # Sidebar
    with st.sidebar:
        st.title("Paramètres")
        gmail_user = st.text_input("Adresse Gmail")
        gmail_pass = st.text_input("Mot de passe", type="password")
        fetch_btn = st.button("Récupérer les emails")
    
    # Main content
    st.title("✉️ Détection de Spam")
    
    # Onglets
    tab1, tab2 = st.tabs(["Détection", "Gmail"])
    
    with tab1:
        st.subheader("Analyser un email")
        email_text = st.text_area("Collez votre email ici", height=200)
        
        if st.button("Vérifier"):
            if email_text.strip():
                label, confidence = classifier.predict(email_text)
                st.markdown(
                    f'<div style="background-color: {classifier.colors[label]}; padding: 10px; border-radius: 5px;">'
                    f'<strong>Résultat:</strong> {classifier.labels[label]} (confiance: {confidence:.0%})'
                    '</div>',
                    unsafe_allow_html=True
                )
            else:
                st.warning("Veuillez entrer un email")
    
    with tab2:
        st.subheader("Emails Gmail")
        if fetch_btn and gmail_user and gmail_pass:
            with st.spinner("Récupération des emails..."):
                emails = fetch_emails(gmail_user, gmail_pass)
                
                if emails:
                    st.success(f"{len(emails)} emails trouvés")
                    for i, email_msg in enumerate(emails):
                        with st.expander(f"Email {i+1}: {email_msg['subject']}"):
                            st.write(f"De: {email_msg['from']} | Date: {email_msg['date']}")
                            
                            full_text = f"{email_msg['subject']}\n{email_msg['body']}"
                            label, confidence = classifier.predict(full_text)
                            
                            st.markdown(
                                f'<div style="background-color: {classifier.colors[label]}; padding: 10px; border-radius: 5px; margin-bottom: 10px;">'
                                f'<strong>Résultat:</strong> {classifier.labels[label]} (confiance: {confidence:.0%})'
                                '</div>',
                                unsafe_allow_html=True
                            )
                            
                            st.text_area("Contenu", value=full_text, height=200, key=f"email_{i}")
                else:
                    st.warning("Aucun email trouvé")
        elif fetch_btn:
            st.warning("Veuillez entrer vos identifiants Gmail")

if __name__ == "__main__":
    main()