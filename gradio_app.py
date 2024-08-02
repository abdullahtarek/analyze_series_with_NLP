import gradio as gr
from text_classifier import JutsuClassifier
from theme_classifier import ThemeClassifier
from character_network import CharacterNetworkGenerator,NamedEntityRecognizer
from character_chat_bot import CharacterChatBot
import os
import dotenv
dotenv.load_dotenv()

character_network = None

def classify_text(model_path,data_path, text):
    jutus_classifer = JutsuClassifier(
        model_path=model_path,
        data_path=data_path,
        huggingface_token=os.getenv("huggingface_token")
        )
    output= jutus_classifer.classify_jutsu([text])
    return output[0] 

def get_themes(theme_list_str,subtitles_path, save_path):
    theme_list = theme_list_str.split(',')
    theme_classfier_cls = ThemeClassifier(theme_list)
    output_df = theme_classfier_cls.get_themes(subtitles_path, save_path)
    
    #remove dialogue from theme_list
    theme_list = [theme for theme in theme_list if theme != 'dialogue']

    output_df  = output_df[theme_list].mean().reset_index()
    output_df.columns = ['Theme','Score']
  
    output_chart = gr.BarPlot(
            output_df,
            x="Theme",
            y="Score",
            title="Seires Themes",
            tooltip=["Theme", "Score"],
            vertical=False,
            width=500,
            height=260,

        )
    return output_chart

def get_character_network(subtitles_path, ner_output_path):
    ner = NamedEntityRecognizer()
    ner_df = ner.get_ners(subtitles_path,ner_output_path)

    character_netowrk_generator = CharacterNetworkGenerator()
    relationship_df = character_netowrk_generator.generate_character_network(ner_df)
    html = character_netowrk_generator.draw_network_graph(relationship_df)

    return html

def alternatingly_agree(message, history):
    # global character_network
    # output = character_network.chat(message,history)
    # return output['content'].strip()
    return "Hello"

def main():
    with gr.Blocks() as iface:
        # Theme Classication with Zero Shot classifiers
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Theme CLassification (Zero Shot Classifers)</h1>")
                with gr.Row():
                    with gr.Column():
                        plot = gr.BarPlot()
                    with gr.Column():
                        theme_list = gr.Textbox(label="Themes")
                        subtitles_path = gr.Textbox(label="Sub titles or Script Path")
                        save_path = gr.Textbox(label="Save Path")
                        get_themes_button = gr.Button("Get Themes")
                        get_themes_button.click(get_themes, inputs=[theme_list,subtitles_path, save_path],outputs=[plot])

        # Character Networks with NERs and Graphs
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Character Network (NERs and Graphs)</h1>")
                with gr.Row():
                    with gr.Column():
                        netwrok_html = gr.HTML()
                    with gr.Column():
                        subtitles_path = gr.Textbox(label="Sub titles or Script Path")
                        ner_path = gr.Textbox(label="NER Save Path")
                        get_network_graph_button = gr.Button("Get Character Network")
                        get_network_graph_button.click(get_character_network, inputs=[subtitles_path,ner_path],outputs=[netwrok_html])

        # Text Classfication with LLMs
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Text Classification With LLMs</h1>")
                with gr.Row():
                    with gr.Column():
                        text_classifcation_output = gr.Textbox(label="Text CLassification Output")
                    with gr.Column():
                        text_classification_model_path = gr.Textbox(label="Model Path")
                        text_classification_data_path = gr.Textbox(label="Data Path")
                        text_to_classify = gr.Textbox(label="Text Input")
                        classify_text_Button = gr.Button("Classify Text (Jutsu)")
                        classify_text_Button.click(classify_text, inputs=[text_classification_model_path, text_classification_data_path, text_to_classify], outputs=[text_classifcation_output])

        # Character Chatbot
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Character Chatbot</h1>")
                gr.ChatInterface(alternatingly_agree)

    iface.launch(share=True)

if __name__ == "__main__":
    main()