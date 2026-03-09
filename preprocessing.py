import email  # for email easy parsing
import re      # for cleaning
import unicodedata  # for unicode


def parse(rawtext):
    msg = email.message_from_string(rawtext)

    # header fields
    subject = msg.get("Subject", "")
    sender = msg.get("From", "")
    reply_to = msg.get("Reply-To", "")
    received = " ".join(msg.get_all("Received", []))
    multitype = msg.get_content_type()

    # body, attachments
    body = ""
    attachment: bool = False
    if msg.is_multipart():
        for part in msg.walk():
            disposition = part.get_content_disposition()
            if disposition == "attachment":
                attachment = True
                continue
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload is not None:
                    body += payload.decode(errors="ignore")
    else:
        payload = msg.get_payload(decode=True)
        if payload is not None:
            body = payload.decode(errors="ignore")

    attachment_word = " attachment" if attachment else ""

    # Combine everything
    combined = f"{subject} {sender} {reply_to} {received} {multitype} {body}{attachment_word}"
    return combined


def clean(text):
    text = unicodedata.normalize("NFKC", text)          # 1. Normalize unicode
    text = text.lower()                                  # 2. Decode and lowercase
    text = re.sub(r'http\S+|www\S+', ' url ', text)     # 3. Flag and remove URLs
    text = re.sub(r'\S+@\S+', ' emailaddr ', text)      # 4. Flag email addresses
    text = re.sub(r'[!?]{2,}', ' multiexclaim ', text)  # 5. Flag excessive punctuation

    # 6. Flag prices/numbers could be spammy
    text = re.sub(r'\$[\d,]+', ' price ', text)
    text = re.sub(r'\d+', ' num ', text)

    # 7. Chinese
    chinese = r'[\u4e00-\u9fff]+'

    def splitchinese(match):
        return " ".join(list(match.group()))  # split into individual characters

    text = re.sub(chinese, splitchinese, text)

    text = re.sub(r'[^\w\s]', ' ', text)       # 8. Remove leftover punctuation (but keep spaces)
    text = re.sub(r'\s+', ' ', text).strip()   # 9. Collapse whitespace

    return text
