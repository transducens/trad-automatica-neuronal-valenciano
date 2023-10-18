import ctranslate2
import sentencepiece as spm

#Modificar el valor de esta variable según el par de idiomas deseado:
# "es-ca": español-valenciano
# "ca-es": valenciano-español
# "en-ca": inglés-valenciano
# "ca-en": valenciano-inglés
pair="ca-es"

sp = spm.SentencePieceProcessor(pair+"/spm/spm.model")
translator = ctranslate2.Translator(pair+"/ct2", device="cpu")

text= "Este reglament s'aprovarà hui amb esta finalitat per la Generalitat Valenciana."

input_tokens = sp.encode(text, out_type=str)
results = translator.translate_batch([input_tokens], beam_size=1, max_batch_size=20, max_decoding_length=len(input_tokens) * 3)
output_tokens = results[0].hypotheses[0]
output_text = sp.decode(output_tokens)

print(output_text)
