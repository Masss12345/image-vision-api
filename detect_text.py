import os
def detect_text(path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS']="C:/Users/MaSsS/Downloads/qwiklabs-gcp-04-d71f5d12165c-2fde894eb2ca.json"
    """Detects text in the file."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print('Texts:')

    words=[]
    for text in texts:
        words.append(text.description)

    
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
        
    return words[0]