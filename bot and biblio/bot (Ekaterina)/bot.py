import logging

import requests
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, filters, MessageHandler, CallbackQueryHandler
import main as m
from using_video_model import YourModule
from text_transformer import RobertaClass


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def build_menu(buttons, n_cols,
               header_buttons=None,
               footer_buttons=None):
    menu = [buttons[i:i + n_cols] for i in range(0, len(buttons), n_cols)]
    if header_buttons:
        menu.insert(0, [header_buttons])
    if footer_buttons:
        menu.append([footer_buttons])
    return menu

# Event that occurs when user send a video to bot
async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    video_file = update.message.video.file_id  # Get ID of video on Telegram server
    video_url = (await context.bot.get_file(video_file)).file_path  # Get URL of video
    video = requests.get(video_url)  # Download the video
    with open('data/video.mp4', 'wb') as f:
        f.write(video.content)  # Write video to file "video.mp4"

    await context.bot.send_message(chat_id=update.effective_chat.id, text="Your video was accepted!")
    await predict(update, context)


async def handle_video_doc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    video_file = update.message.document.file_id  # Get ID of video on Telegram server
    video_url = (await context.bot.get_file(video_file)).file_path  # Get URL of video
    video = requests.get(video_url)  # Download the video
    with open('data/video.mp4', 'wb') as f:
        f.write(video.content)  # Write video to file "video.mp4"

    await context.bot.send_message(chat_id=update.effective_chat.id, text="Your video was accepted!")
    await predict(update, context)

async def button(update, _):
    query = update.callback_query
    variant = query.data
    await query.answer()
    if (variant == '1'):
        pred, texts = m.start_predict('latest_fusion', 'data/video.mp4')
        await query.edit_message_text(text=f"Late fusion: {pred} \n Transcription: {texts}")
    if (variant == '2'):
        pred, texts = m.start_predict('early_fusion', 'data/video.mp4')
        await query.edit_message_text(text=f"Early fusion: {pred}\n Transcription: {texts}")
    if (variant == '3'):
        await query.edit_message_text(text="Please, upload your video")



async def predict(update, context: ContextTypes.DEFAULT_TYPE):
    button_list = [
        InlineKeyboardButton("late fusion", callback_data='1'),
        InlineKeyboardButton("early fusion", callback_data='2'),
    ]
    reply_markup = InlineKeyboardMarkup(build_menu(button_list, n_cols=2))
    await context.bot.send_message(chat_id=update.effective_chat.id,
                                   text="Which prediction would you like to use?", reply_markup=reply_markup)


# When press "start" button
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    button_list = [
        InlineKeyboardButton("Upload video", callback_data='3')
    ]
    reply_markup = InlineKeyboardMarkup(build_menu(button_list, n_cols=2))
    await context.bot.send_message(chat_id=update.effective_chat.id, text="I'm a bot, which can predict emotions by videos with utterance!"
                                                                          " Please, send me one and I'll show you!", reply_markup=reply_markup)

if __name__ == '__main__':
    # Insert the token
    application = ApplicationBuilder().token('6197842128:AAHYLOmWzIuEgRdDuFXweSK__LkC1y_AomE').build()

    # Set function to "start" command
    start_handler = CommandHandler('start', start)
    application.add_handler(start_handler)

    # Set function to video handler
    video_handler = MessageHandler(filters.VIDEO, handle_video)
    application.add_handler(video_handler)

    video_handler2 = MessageHandler(filters.Document.MP4, handle_video_doc)
    application.add_handler(video_handler2)

    predict_handler = CommandHandler('predict', predict)
    application.add_handler(predict_handler)
    application.add_handler(CallbackQueryHandler(button))

    # Run the bot
    application.run_polling()
