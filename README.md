# AI YouTube Shorts Generator

Превращает одно длинное видео (ссылка YouTube, локальный файл или папка) в
ранжированные вертикальные шортсы **9:16** — полностью локально, без водяных
знаков и без лимитов на минуты. `yt-dlp` скачивает исходник, `faster-whisper`
распознаёт речь, LLM (OpenAI / DeepSeek / Gemini) ранжирует самые «вирусные»
моменты, а `ffmpeg` + OpenCV кадрируют их вертикально. Опционально можно добавить
**фоновую музыку**, **вшить субтитры**, довести кадр до **9:16** с размытым фоном
и наложить **баннер** (картинку или текстовую полосу).

> Бесплатная альтернатива Opus Clip / Vidyo.ai / Klap.

## Установка

```bash
git clone https://github.com/SamurAIGPT/AI-Youtube-Shorts-Generator.git
cd AI-Youtube-Shorts-Generator
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # Windows: copy .env.example .env
```

Требования: Python 3.10+, `ffmpeg` в `PATH` и один ключ LLM.

## Настройка

Всё живёт в одном файле `.env` (скопируйте из `.env.example`). Самое главное:

```env
INPUT=video/talk.mkv      # файл, папка с видео или ссылка на YouTube
OUTPUT_DIR=output         # куда писать результат
MUSIC=music               # файл с музыкой или папка (необязательно)
```

Провайдер выбирается через `LLM_PROVIDER`, его ключ — `OPENAI_API_KEY`,
`DEEPSEEK_API_KEY` или `GEMINI_API_KEY`. Остальные ключи (Whisper, вид субтитров,
кодирование) описаны в `.env.example`. Любой флаг команды перекрывает `.env` на
один запуск.

## Команды

```bash
python main.py clip         # нарезать исходник в ранжированные вертикальные шортсы
python main.py transcribe   # Whisper -> <видео>.srt (файл или папка)
python main.py music        # добавить музыку (MUSIC: файл или папка)
python main.py subtitles    # вшить субтитры (из .srt или через Whisper)
python main.py all          # нарезка + музыка + субтитры одной командой
```

Каждая команда читает `INPUT` / `OUTPUT_DIR` из `.env` и принимает:

| Флаг | Значение |
|---|---|
| `-i, --input` | видео, папка или ссылка (по умолчанию `INPUT`) |
| `-o, --output-dir` | куда писать результат (по умолчанию `OUTPUT_DIR`) |
| `-q, --quiet` | только предупреждения и ошибки |
| `--env FILE` | дополнительный `.env` (наивысший приоритет) |

### Примеры

```bash
python main.py clip -i "video/talk.mkv" -n 5 -a 9:16 -l ru
python main.py transcribe -i video/            # все видео в папке
python main.py music -m music/                 # случайный трек из папки
python main.py music -m song.mp3               # конкретный файл
python main.py subtitles                       # одноимённый .srt, иначе Whisper
python main.py subtitles --banner "@mychannel"   # вшить субтитры + текстовый баннер
python main.py subtitles --banner assets/logo.png --banner-position top  # баннер-картинка
python main.py all --banner "@mychannel" --banner-position bottom   # весь пайплайн + баннер
python main.py all                             # весь пайплайн из .env
```

- **clip**: `-n/--num-clips`, `-a/--aspect-ratio`, `--format`, `-l/--language`,
  `--[no-]face-tracking`, `--output-json`.
- **subtitles** (`-s/--source`, по умолчанию `auto`): если рядом с видео лежит
  одноимённый `.srt` — берётся он, иначе сначала идёт распознавание. `file` —
  только готовый файл (иначе ошибка), `whisper` — всегда распознавать, `none` —
  не трогать субтитры.
- **music**: если `MUSIC` — папка, берётся случайный трек, если файл — именно он.
  Музыка подмешивается *под* голос, поэтому дорожка речи не теряется.
- **вертикальный кадр и баннер** (`subtitles`, `music`, `all`):
  `--[no-]fit-vertical`, `--fit-aspect-ratio W:H`, `--background-blur N`,
  `--banner TEXT|PATH`, `--banner-position {top,bottom,center}`.

## Субтитры

Субтитры вшиваются короткими репликами: длинное предложение Whisper нарезается
на отдельные фразы, поэтому на экране всегда висит одна короткая фраза, а не
«простыня» на пол-экрана на десять секунд.

- Реплики строятся по таймингам **отдельных слов** (`word_timestamps`), поэтому
  текст появляется и исчезает синхронно с голосом. Новая реплика начинается на
  конце предложения, на заметной паузе в речи или после запятой/тире, если фраза
  уже достаточно длинная.
- Длину реплики ограничивают три настройки в `.env`:
  - `SUBTITLE_MAX_CHARS` — максимум символов в реплике (по умолчанию `40`);
  - `SUBTITLE_MAX_WORDS` — максимум слов (по умолчанию `9`);
  - `SUBTITLE_MAX_DURATION` — максимум секунд на экране (по умолчанию `3.5`).
- Тот же разрез применяется и к готовому `.srt` (старый кэш или свой файл):
  перед вшиванием длинные строки дробятся на короткие фразы.
- **Синхронность с речью** — `SUBTITLE_OFFSET` (в секундах, по умолчанию `0.0`).
  Тайминги отдельных слов Whisper иногда «забегают» вперёд реальной речи, из-за
  чего реплика появляется на долю секунды раньше фразы. Положительное значение
  сдвигает показ **позже** (например `SUBTITLE_OFFSET=0.2`), отрицательное —
  раньше. Сдвигается только показ субтитров на видео; сам кэш `.srt` сохраняет
  исходные тайминги.
- **Анимация появления** реплики — `SUBTITLE_ANIMATION` (пусто = без анимации,
  по умолчанию). Доступные значения:
  - `fade` — реплика просто проявляется;
  - `slide` — проявляется, всплывая снизу вверх;
  - `pop` — проявляется, увеличиваясь с 70% до 100%.
  Длительность входа задаёт `SUBTITLE_ANIMATION_DURATION` (в секундах, по
  умолчанию `0.25`). Анимация проигрывается только в начале реплики, дальше
  текст стоит на месте. Пустое значение `SUBTITLE_ANIMATION` (а также
  `none`/`off`) отключает анимацию полностью.
- Кэш `.srt` создаётся уже нарезанным на реплики. Если рядом с видео лежит
  **старый** `.srt` (одна строка = одно длинное предложение), удалите его один
  раз — при следующем запуске `transcribe` / `subtitles` / `all` текст
  распознается заново уже с таймингами по словам.

## Вертикальный кадр и баннер

Эти возможности применяются в постобработке — в том же шаге, где вшиваются
субтитры (команды `subtitles`, `music` и `all`).

- **Вертикальный кадр** (`FIT_VERTICAL=true`, включён по умолчанию). Если видео
  после нарезки не совпадает по пропорциям с `FIT_ASPECT_RATIO` (по умолчанию
  `9:16`), оно вписывается по центру, а пустые места закрывает размытая копия
  этого же видео. Уже вертикальные видео не пересобираются. Настройки:
  `FIT_ASPECT_RATIO` (пусто = взять `ASPECT_RATIO`), `FIT_HEIGHT` (высота кадра,
  по умолчанию 1920), `BACKGROUND_BLUR` (сила размытия, 0 = без размытия),
  `BACKGROUND_DARKEN` (затемнение фона, 0..1).
- **Баннер** (`BANNER`). Один аргумент: если значение указывает на существующий
  файл — рисуется картинкой (`PNG` с прозрачностью или `JPG`), иначе само значение
  рисуется текстовой полосой во всю ширину кадра. Настройки: `BANNER_POSITION`
  (`top`/`bottom`/`center`), `BANNER_WIDTH_RATIO`, `BANNER_OPACITY`,
  `BANNER_MARGIN`, `BANNER_FONT_SIZE`, `BANNER_TEXT_COLOR`,
  `BANNER_BACKGROUND_COLOR`.

Примеры:

```bash
# текстовый баннер во всю ширину кадра (полоса с текстом сверху)
python main.py subtitles -i "video/talk.mkv" --banner "@mychannel"

# баннер-картинка (PNG с прозрачностью) внизу кадра
python main.py subtitles -i "video/talk.mkv" --banner "assets/logo.png" --banner-position bottom

# баннер в полном пайплайне: нарезка + музыка + субтитры + баннер 9:16
python main.py all --banner "@mychannel" --banner-position bottom

# баннер из .env (одна переменная BANNER: путь к файлу или текст)
python main.py subtitles

# без баннера, оставить исходные пропорции
python main.py subtitles -i clip.mp4 --no-fit-vertical
```

## Как это работает

1. **Скачивание** — `yt-dlp` (локальный файл берётся как есть).
2. **Распознавание** — `faster-whisper` (с таймингами по отдельным словам),
   результат кэшируется в `.srt` и сразу режется на короткие реплики-субтитры.
3. **Ранжирование** — LLM оценивает каждый момент 0–100 по «вирусной» рамке
   (хуки, эмоциональные пики, мнения-бомбы, откровения, конфликт, цитаты, пики
   историй, практическая польза).
4. **Дедупликация и отбор** — пересекающиеся кандидаты схлопываются, остаётся топ
   `NUM_CLIPS`.
5. **Нарезка и кроп** — `ffmpeg` режет, OpenCV трекает лицо для вертикального кадра.
6. **Постобработка** *(опционально)* — кадр доводится до 9:16 (пустые места
   закрывает размытая копия видео), накладывается баннер, вшиваются субтитры,
   под звук подмешивается музыка.

## Python API

```python
from shorts_generator import load_settings, generate_shorts

settings = load_settings(extra={"input": "URL", "num_clips": 5})
result = generate_shorts(settings)            # enhance=True добавит музыку + субтитры
for short in result["shorts"]:
    print(short["score"], short["title"], short["clip_url"])
```

## Структура проекта

```
main.py                     точка входа CLI
.env.example                единый шаблон настроек
shorts_generator/
├── cli.py                  подкоманды (clip / transcribe / music / subtitles / all)
├── config.py               единый загрузчик .env -> Settings
├── pipeline.py             оркестратор + публичный API
├── downloader.py           скачивание через yt-dlp
├── transcriber.py          faster-whisper (+ кэш .srt)
├── highlights.py           ранжирование моментов через LLM
├── llm.py                  бэкенды OpenAI / DeepSeek / Gemini
├── clipper.py              нарезка ffmpeg + вертикальный кроп OpenCV
├── subtitles.py            Whisper -> .srt
├── cues.py                 нарезка транскрипта на короткие реплики-субтитры
├── enhance.py              музыка + субтитры на каждый клип (команда all)
└── postprocess/            движок вшивания (субтитры + музыка + баннер + вертикальный кадр)
```

## Устранение неполадок

- **Whisper не нашёл речи** — нет речи или не тот язык; задайте `WHISPER_LANGUAGE`.
- **Музыка не добавилась** — проверьте, что `MUSIC` указывает на существующий файл
  или папку.
- **Предупреждение про шрифт** — `FONT` не найден; задайте `FONT` (путь к
  файлу-шрифту, имя файла или папку со шрифтами) или положите `.ttf`/`.ttc` в
  `fonts/`.
- **`ffmpeg` не найден** — установите его или задайте `FFMPEG_PATH`.
- **Предупреждение «could not open … directly; retrying on a cleaned copy»** — во
  входном видео есть метаданные глав, на которых спотыкается парсер MoviePy 2.x
  (он падает на файлах ровно с одной главой). Инструмент сам делает очищенную
  копию без глав и лишних потоков и продолжает работу — на результат это не
  влияет. Копия создаётся только при необходимости и удаляется автоматически.
- **Хочется лучших моментов** — настройте `VIRALITY_CRITERIA` /
  `HIGHLIGHT_SYSTEM_PROMPT` в `shorts_generator/highlights.py` или смените
  `LLM_PROVIDER`.

## Лицензия

MIT. Основано на открытом проекте
[AI-Youtube-Shorts-Generator](https://github.com/SamurAIGPT/AI-Youtube-Shorts-Generator).