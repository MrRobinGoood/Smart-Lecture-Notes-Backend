from aiogram import Bot, Dispatcher, executor, types
import json
from audio_recognition import get_text

token = '6902499601:AAHmJq8-ULNFX8oma8rjCDcy4G7uV6PP1Dg'

bot = Bot(token=token)
dp = Dispatcher(bot)

password = 'топ3'


@dp.message_handler(commands=['start'])
async def starting(message: types.Message):
    await message.answer("Здравствуйте, пожалуйста, введите <b>пароль</b>!\n<i>(Команда Алгоритмы)</i>",
                         parse_mode='html')


@dp.message_handler()
async def password_check(message: types.Message):
    if message.text == password:
        await message.answer(
            "Добро пожаловать в бота <b>Смарт-методист</b> для компании GeekBrains!\nЧтобы начать работу, запишите <b>голосовое сообщение</b> или пришлите мне аудио файл <b>в формате MP3</b> для расшифровки.\nОтвет бота будет в несколько этапов:\n1. Текстовая расшифровка аудио-фрагмента\n2. Файл с составленным конспектом по аудио-фрагменту<i>(Большие файлы могут обрабатываться достаточно долго)</i>",
            parse_mode='html')
        user_id = message.from_user.id
        with open('users_id.json', 'r') as file:
            data = json.load(file)
        if user_id not in data['admins']:
            data['admins'].append(user_id)
            with open('users_id.json', 'w') as file:
                json.dump(data, file, indent=4)


@dp.message_handler(content_types=["voice", "audio"])
async def answer_audio(message: types.Message):
    user_id = message.from_user.id
    with open('users_id.json', 'r') as file:
        data = json.load(file)
    if user_id in data['admins']:
        if message.voice or message.audio:
            await message.reply('Файл принят в обработку✅Ожидайте...')
            if message.voice:
                file_id = message.voice.file_id
            else:
                file_id = message.audio.file_id
            file_info = await bot.get_file(file_id)
            file = await bot.download_file_by_id(file_info.file_id)
            result, rec_time = get_text(file.getvalue())
            await message.reply(f'Обработка⚙1/3!\n<b>Расшифровка аудио-фрагмента:</b>\n{result["text"].strip()}',
                                parse_mode='html')


if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
