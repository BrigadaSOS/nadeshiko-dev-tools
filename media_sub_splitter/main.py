import concurrent
import argparse
import threading
import csv
import json
import logging
import os
import pathlib
import re
import shutil
import string
import subprocess
from collections import namedtuple
from datetime import timedelta
from datetime import date
from pathlib import Path
from multiprocessing.pool import ThreadPool as Pool
from imdb import Cinemagoer
import babelfish
import deepl
import ffmpeg
import inquirer
import jaconvV2
import moviepy.editor as mp
import pysubs2
import requests
from anilist import Client
from langdetect import detect
from dotenv import load_dotenv
from guessit import guessit
from themoviedb import TMDb

logging.getLogger("moviepy").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
logger.propagate = 0
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)-15s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

emoji = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "]+",
    re.UNICODE,
)

SUPPORTED_LANGUAGES = ["en", "ja", "es"]

EpisodeTsvRow = namedtuple(
    "Row",
    [
        "ID",
        "SUBS_JP_IDS",
        "SUBS_ES_IDS",
        "SUBS_EN_IDS",
        "START_TIME",
        "END_TIME",
        "NAME_AUDIO",
        "NAME_SCREENSHOT",
        "CONTENT",
        "CONTENT_TRANSLATION_SPANISH",
        "CONTENT_TRANSLATION_ENGLISH",
        "CONTENT_SPANISH_MT",
        "CONTENT_ENGLISH_MT",
        "ACTOR_JA",
        "ACTOR_ES",
        "ACTOR_EN",
    ],
)

MatchingSubtitle = namedtuple("MatchingSubtitle", ["origin", "data", "filepath"])



# Audiobooks
AudiobookTsvRow = namedtuple(
    "AudiobookRow",
    [
        "ID",
        "CHAPTER_NUM",
        "START_TIME",
        "END_TIME",
        "NAME_AUDIO",
        "NAME_SCREENSHOT",
        "CONTENT",
        "CONTENT_TRANSLATION_SPANISH",
        "CONTENT_TRANSLATION_ENGLISH",
        "CONTENT_SPANISH_MT",
        "CONTENT_ENGLISH_MT",
    ],
)


def merge_audiobook_subtitles(
    processed_subtitles, 
    max_gap_ms=700, 
    max_duration_ms=8000,  # 8s por segmento
    pad_ms=1, 
    dedup_adjacent=True
):
    """
    Une líneas si están cerca o solapadas, pero corta:
      - si hay un gap grande
      - si la duración supera max_duration_ms
      - si hay un punto aparte (fin de oración)
    """
    if not processed_subtitles:
        return []

    lines = sorted(processed_subtitles, key=lambda x: (x["start"], x["end"]))

    merged = []
    cur = {
        "ids": [lines[0]["id"]],
        "start": lines[0]["start"],
        "end": lines[0]["end"],
        "texts": [lines[0]["text"]],
        "original_texts": [lines[0].get("original_text", lines[0]["text"])],
    }
    last_text = lines[0]["text"]
    last_end = lines[0]["end"]

    sentence_end_re = re.compile(r"[。．\.!?！？]\s*$")  # signos de final de oración

    for line in lines[1:]:
        start, end, text = line["start"], line["end"], line["text"]
        original_text = line.get("original_text", text)

        overlap = (cur["start"] - pad_ms) < end and start < (cur["end"] + pad_ms)
        close_enough = (start - cur["end"]) <= max_gap_ms
        duration = end - cur["start"]

        # Condiciones de corte
        too_long = duration > max_duration_ms
        prev_ends_sentence = bool(sentence_end_re.search(cur["texts"][-1]))

        if (overlap or close_enough) and not too_long and not prev_ends_sentence:
            # unir
            if not (dedup_adjacent and last_text == text and last_end == start):
                cur["texts"].append(text)
                cur["original_texts"].append(original_text)
            cur["ids"].append(line["id"])
            cur["end"] = max(cur["end"], end)
            last_text = text
            last_end = end
        else:
            # cortar y guardar
            merged.append({
                "id": cur["ids"][0],
                "start": cur["start"],
                "end": cur["end"],
                "text": " ".join(cur["texts"]).strip(),
                "original_text": " ".join(cur["original_texts"]).strip(),
            })
            cur = {
                "ids": [line["id"]],
                "start": start,
                "end": end,
                "texts": [text],
                "original_texts": [original_text],
            }
            last_text = text
            last_end = end

    # último segmento
    merged.append({
        "id": cur["ids"][0],
        "start": cur["start"],
        "end": cur["end"],
        "text": " ".join(cur["texts"]).strip(),
        "original_text": " ".join(cur["original_texts"]).strip(),
    })

    # reindexar
    for i, seg in enumerate(merged):
        seg["id"] = i

    return merged

def extract_segments_from_audiobook(
    audiobook_folder,
    output_folder,
    translator,
    args
):
    """
    Procesa una carpeta de audiolibro que contiene:
    - archivo.mp3
    - archivo.srt
    - cover.jpg
    - chapters.txt
    """
    try:
        logger.info(f"Processing audiobook folder: {audiobook_folder}")
        
        # Buscar archivos necesarios
        mp3_files = [f for f in os.listdir(audiobook_folder) if f.endswith('.mp3')]
        srt_files = [f for f in os.listdir(audiobook_folder) if f.endswith('.srt')]
        cover_files = [f for f in os.listdir(audiobook_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        chapters_files = [f for f in os.listdir(audiobook_folder) if f.lower() == 'chapters.txt']
        
        if not mp3_files:
            logger.error("No se encontró archivo MP3 en la carpeta")
            return
        
        if not srt_files:
            logger.error("No se encontró archivo SRT en la carpeta")
            return
            
        # Usar el primer archivo encontrado de cada tipo
        mp3_file = os.path.join(audiobook_folder, mp3_files[0])
        srt_file = os.path.join(audiobook_folder, srt_files[0])
        cover_file = os.path.join(audiobook_folder, cover_files[0]) if cover_files else None
        chapters_file = os.path.join(audiobook_folder, chapters_files[0]) if chapters_files else None
        
        # Extraer título del nombre de la carpeta
        audiobook_title = os.path.basename(audiobook_folder)
        
        # Crear carpeta de salida para el audiolibro
        audiobook_folder_name = map_audiobook_title_to_folder(audiobook_title)
        audiobook_output_path = os.path.join(output_folder, audiobook_folder_name)
        os.makedirs(audiobook_output_path, exist_ok=True)
        
        # Crear info.json para el audiolibro
        info_json_path = os.path.join(audiobook_output_path, "info.json")
        if not os.path.exists(info_json_path):
            logger.info("Creando info.json para audiolibro...")
            
            # Cargar información de capítulos si existe
            chapters_info = load_chapters_info(chapters_file) if chapters_file else []
            
            info_json = {
                "version": "5",
                "type": "audiobook",
                "title": audiobook_title,
                "folder_name": audiobook_folder_name
            }
            
            # Copiar cover si existe
            if cover_file and os.path.exists(cover_file):
                cover_dest = os.path.join(audiobook_output_path, "cover.jpg")
                shutil.copy2(cover_file, cover_dest)
                info_json["cover"] = "cover.jpg"
            
            # Guardar info.json
            with open(info_json_path, "w", encoding="utf-8") as f:
                json.dump(info_json, f, indent=2, ensure_ascii=False)
        
        # Procesar subtítulos
        logger.info("Cargando subtítulos...")
        subtitles = pysubs2.load(srt_file)
        
        # Detectar idioma de los subtítulos
        subtitle_text = " ".join([event.text for event in subtitles[:10]])  # Usar primeras 10 líneas
        detected_language = detect(subtitle_text)
        logger.info(f"Idioma detectado en subtítulos: {detected_language}")
        
        # Crear carpeta para segmentos
        segments_folder = os.path.join(audiobook_output_path, "segments")
        os.makedirs(segments_folder, exist_ok=True)
        
        # Procesar segmentos
        split_audiobook_by_subtitles(
            translator,
            mp3_file,
            subtitles,
            segments_folder,
            detected_language,
            chapters_file,
            args
        )
        
        logger.info(f"Audiolibro procesado exitosamente: {audiobook_title}")
        
    except Exception as e:
        logger.error(f"Error procesando audiolibro: {e}", exc_info=True)

def load_chapters_info(chapters_file):
    """Cargar información de capítulos desde chapters.txt"""
    chapters = []
    if not chapters_file or not os.path.exists(chapters_file):
        return chapters
    
    try:
        with open(chapters_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    # Formato esperado: "00:00:00 Capítulo 1"
                    # o simplemente "Capítulo 1" si no hay timestamps
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        time_str, chapter_name = parts
                        if re.match(r'\d{1,2}:\d{2}:\d{2}', time_str):
                            chapters.append({
                                "number": line_num,
                                "name": chapter_name,
                                "timestamp": time_str
                            })
                        else:
                            chapters.append({
                                "number": line_num,
                                "name": line,
                                "timestamp": None
                            })
                    else:
                        chapters.append({
                            "number": line_num,
                            "name": line,
                            "timestamp": None
                        })
    except Exception as e:
        logger.warning(f"Error cargando capítulos: {e}")
    
    return chapters

def split_audiobook_by_subtitles(
    translator,
    audio_file,
    subtitles,
    output_folder,
    subtitle_language,
    chapters_file,
    args,
    output_tsv_name="data.tsv"
):
    """Dividir audiolibro en segmentos basándose en subtítulos (con fusión de líneas cercanas)."""

    audio_clip = mp.AudioFileClip(audio_file)
    chapters_info = load_chapters_info(chapters_file) if chapters_file else []
    tsv_filepath = os.path.join(output_folder, output_tsv_name)
    temp_tsv_filepath = os.path.join(output_folder, "data_temp.tsv")
    # Nuevo: Cargar el TSV existente si lo hay
    
    
    source_tsv_path = tsv_filepath
    if os.path.exists(temp_tsv_filepath):
        logger.info(f"Temporary file '{temp_tsv_filepath}' found. Resuming from last interruption.")
        source_tsv_path = temp_tsv_filepath

    existing_translations = {}
    if os.path.exists(source_tsv_path):
        logger.info(f"Existing TSV file found at {source_tsv_path}. Loading for resumption...")
        with open(source_tsv_path, "r", newline="", encoding="utf-8") as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter="\t")
            for row in reader:
                # Almacenar las traducciones existentes por ID
                if row.get("ID"): # Asegurarse de que el ID exista
                    existing_translations[int(row["ID"])] = {
                        "CONTENT_TRANSLATION_SPANISH": row["CONTENT_TRANSLATION_SPANISH"],
                        "CONTENT_TRANSLATION_ENGLISH": row["CONTENT_TRANSLATION_ENGLISH"],
                        "CONTENT_SPANISH_MT": row["CONTENT_SPANISH_MT"],
                        "CONTENT_ENGLISH_MT": row["CONTENT_ENGLISH_MT"],
                    }

    # 1) Normalizar/filtrar subtítulos
    processed_subtitles = []
    for i, subtitle in enumerate(subtitles):
        if subtitle.text and subtitle.text.strip():
            processed_text = process_audiobook_subtitle(subtitle.text, args)
            if processed_text:
                processed_subtitles.append({
                    "id": i,
                    "start": subtitle.start,  # ms
                    "end": subtitle.end,      # ms
                    "text": processed_text,
                    "original_text": subtitle.text
                })

    # 2) Fusionar segmentos por proximidad (usa 1100ms por defecto o args.merge_gap_ms si existe)
    max_gap_ms = getattr(args, "merge_gap_ms", 700)
    merged_subtitles = merge_audiobook_subtitles(processed_subtitles, max_gap_ms=max_gap_ms, pad_ms=1, dedup_adjacent=True)

    # 3) Emitir TSV y audios
    # 3) Escribir en un nuevo TSV, utilizando las traducciones existentes
    temp_tsv_filepath = os.path.join(output_folder, "data_temp.tsv")
    with open(temp_tsv_filepath, "w", newline="", encoding="utf-8") as tsvfile:
        writer = csv.DictWriter(
            tsvfile,
            fieldnames=AudiobookTsvRow._fields,
            delimiter="\t",
            quoting=csv.QUOTE_NONE,
            escapechar="\\"
        )
        writer.writeheader()

        for subtitle_data in merged_subtitles:
            segment_id = subtitle_data["id"]
            
            # Nuevo: Reusar traducciones existentes si ya están hechas
            if segment_id in existing_translations:
                # Usar los datos existentes
                existing_data = existing_translations[segment_id]
                subtitle_data["translation_spanish"] = existing_data["CONTENT_TRANSLATION_SPANISH"]
                subtitle_data["translation_english"] = existing_data["CONTENT_TRANSLATION_ENGLISH"]
                subtitle_data["spanish_is_mt"] = existing_data["CONTENT_SPANISH_MT"]
                subtitle_data["english_is_mt"] = existing_data["CONTENT_ENGLISH_MT"]
            else:
                # Traducir si no hay datos
                subtitle_data["translation_spanish"] = None
                subtitle_data["translation_english"] = None
                subtitle_data["spanish_is_mt"] = None
                subtitle_data["english_is_mt"] = None

            generate_audiobook_segment(
                subtitle_data,
                audio_clip,
                output_folder,
                translator,
                subtitle_language,
                chapters_info,
                writer,
                args
            )

    audio_clip.close()
    
    # Nuevo: Reemplazar el archivo TSV original con el nuevo
    os.replace(temp_tsv_filepath, tsv_filepath)
    
    logger.info(f"Audiolibro procesado exitosamente: {output_folder}")



def generate_audiobook_segment(
    subtitle_data,
    audio_clip,
    output_folder,
    translator,
    subtitle_language,
    chapters_info,
    writer,
    args
):
    """Generar un segmento individual del audiolibro"""
    
    segment_id = subtitle_data["id"]
    start_ms = subtitle_data["start"]
    end_ms = subtitle_data["end"]
    text = subtitle_data["text"]
    
    # Convertir a segundos
    start_seconds = start_ms / 1000.0
    end_seconds = end_ms / 1000.0
    
    # Determinar capítulo
    chapter_num = determine_chapter(start_ms, chapters_info)
    
    # Nombres de archivos
    audio_filename = f"{segment_id:06d}.mp3"
    
    # Generar audio del segmento (opcional si ya existe)
    audio_path = os.path.join(output_folder, audio_filename)
    if not os.path.exists(audio_path) and not args.dryrun:
        try:
            audio_segment = audio_clip.subclip(start_seconds, end_seconds)
            audio_segment.write_audiofile(audio_path, codec="mp3", logger=None)
            # logger.info(f"Segmento de audio guardado: {text} - {audio_path} ({start_seconds}-{end_seconds})")
        except Exception as e:
            logger.error(f"Error generando audio para segmento {segment_id}: {e}")
            return
    
    # Traducciones
    text_spanish = subtitle_data.get("translation_spanish")
    text_english = subtitle_data.get("translation_english")
    spanish_is_mt = subtitle_data.get("spanish_is_mt")
    english_is_mt = subtitle_data.get("english_is_mt")
    
    # Nuevo: Solo traducir si las traducciones no existen
    if not text_spanish and translator:
        try:
            if subtitle_language != "es":
                text_spanish = translator.translate_text(
                    text, 
                    source_lang=subtitle_language.upper(), 
                    target_lang="ES"
                ).text
                spanish_is_mt = "True"
        except Exception as e:
            logger.warning(f"Error traduciendo segmento {segment_id} al español: {e}")
            text_spanish = "" # Evitar rehacer la traducción en el futuro
    
    if not text_english and translator:
        try:
            if subtitle_language != "en":
                text_english = translator.translate_text(
                    text, 
                    source_lang=subtitle_language.upper(), 
                    target_lang="EN-US"
                ).text
                english_is_mt = "True"
        except Exception as e:
            logger.warning(f"Error traduciendo segmento {segment_id} al inglés: {e}")
            text_english = "" # Evitar rehacer la traducción en el futuro
            
    # Asignar texto original si el idioma es el mismo
    if subtitle_language == "es":
        text_spanish = text
        spanish_is_mt = "False"
    elif subtitle_language == "en":
        text_english = text
        english_is_mt = "False"

    logger.info(f"ID: {segment_id} | File: {audio_filename} | Text: '{text}' | MTS: '{text_spanish}' | MTE: '{text_english}'")
    
    # Escribir fila en TSV
    writer.writerow(
        AudiobookTsvRow(
            ID=segment_id,
            CHAPTER_NUM=chapter_num,
            START_TIME=timedelta(milliseconds=start_ms),
            END_TIME=timedelta(milliseconds=end_ms),
            NAME_AUDIO=audio_filename,
            NAME_SCREENSHOT="", 
            CONTENT=text,
            CONTENT_TRANSLATION_SPANISH=text_spanish,
            CONTENT_TRANSLATION_ENGLISH=text_english,
            CONTENT_SPANISH_MT=spanish_is_mt,
            CONTENT_ENGLISH_MT=english_is_mt,
        )._asdict()
    )

def determine_chapter(timestamp_ms, chapters_info):
    """Determinar a qué capítulo pertenece un timestamp"""
    if not chapters_info:
        return 1
    
    for i, chapter in enumerate(chapters_info):
        if chapter.get("timestamp"):
            # Convertir timestamp del capítulo a millisegundos
            time_parts = chapter["timestamp"].split(":")
            chapter_ms = (int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])) * 1000
            
            # Si encontramos un capítulo que empieza después de nuestro timestamp
            if chapter_ms > timestamp_ms:
                return max(1, i)  # Capítulo anterior
    
    return len(chapters_info)  # Último capítulo

def process_audiobook_subtitle(text, args):
    """Procesar texto de subtítulos para audiolibros"""
    # Limpiar texto básico
    processed_text = text.strip()
    
    # Remover tags HTML comunes
    processed_text = re.sub(r'<[^>]+>', '', processed_text)
    
    # Remover caracteres especiales de control
    processed_text = re.sub(r'[\r\n\t]+', ' ', processed_text)
    
    # Remover espacios múltiples
    processed_text = re.sub(r'\s+', ' ', processed_text)
    
    return processed_text.strip()

def map_audiobook_title_to_folder(title):
    """Mapear título de audiolibro a nombre de carpeta"""
    return "-".join(
        title.lower().translate(str.maketrans("", "", string.punctuation)).split()
    )

#####
def main():
    load_dotenv()
    args = command_args()
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    deepl_token = os.getenv("TOKEN") or args.token
    if not deepl_token:
        logger.warning(
            " > IMPORTANT < DEEPL TOKEN has not been detected. Subtitles won't be translated to all supported languages"
        )

    translator = deepl.Translator(deepl_token) if deepl_token else None

    # Input and output folders
    input_folder = args.input
    output_folder = args.output

    # Selección de tipo de media
    type_media_selection = [
        inquirer.List(
            "media_selection",
            message="What kind of media are you using?",
            choices=['Anime', 'JDrama', 'Audiobook']
        )
    ]
    selected_type_media_selection = inquirer.prompt(type_media_selection)
    
    if selected_type_media_selection["media_selection"] == "Audiobook":
        # Buscar carpetas de audiolibros
        audiobook_folders = [
            os.path.join(input_folder, folder_name)
            for folder_name in os.listdir(input_folder)
            if os.path.isdir(os.path.join(input_folder, folder_name))
        ]
        
        if not audiobook_folders:
            logger.error(f"No se encontraron carpetas de audiolibros en {input_folder}")
            return
        
        logger.info(f"Encontradas {len(audiobook_folders)} carpetas de audiolibros")
        
        for audiobook_folder in audiobook_folders:
            extract_segments_from_audiobook(
                audiobook_folder,
                output_folder,
                translator,
                args
            )
    else:
        # Código existente para anime/jdrama...
        episode_filepaths = sorted([
            os.path.join(root, name)
            for root, dirs, files in os.walk(input_folder)
            for name in files
            if name.endswith(".mkv")
        ])

        if not episode_filepaths:
            logger.error(f"No .mkv files found in {input_folder}! Nothing else to do.")
            return

        logger.info(f"Found {len(episode_filepaths)} files to process in {input_folder}...")

        media_info = CachedMediaInfo()
        subtitles_dict_remembered = {}
        pool = Pool(2)
        
        for episode_filepath in episode_filepaths:
            pool, subtitles_dict_remembered = extract_segments_from_episode(
                pool,
                episode_filepath,
                output_folder,
                translator,
                media_info,
                subtitles_dict_remembered,
                args,
                selected_type_media_selection
            )

        pool.close()
        pool.join()

def url_clean(url_fragment):
    if url_fragment:
        return f"https://image.tmdb.org/t/p/original{url_fragment}"
    return None

def extract_segments_from_episode(
    pool,
    episode_filepath,
    output_folder,
    translator,
    anilist,
    subtitles_dict_remembered,
    args,
    selected_type_media_selection
):
    try:
        logger.info(f"Filepath: {episode_filepath}\n")

        # Guessit
        guessit_query = extract_anime_title_for_guessit(episode_filepath)
        logger.info(f"> Query for Guessit: {guessit_query}")
        episode_info = guessit(guessit_query)

        guessed_anime_title = episode_info["title"]
        season_number_pretty = f"S{episode_info['season']:02d}"
        episode_number_pretty = f"E{episode_info['episode']:02d}"
        logger.info(
            f"Guessed information: {guessed_anime_title} {season_number_pretty}{episode_number_pretty}\n"
        )

        anilist_query = extract_anime_title_for_anilist(guessed_anime_title)
        logger.info(f"Query for media: {anilist_query}")
        anime_info = anilist.get_media_info(anilist_query, selected_type_media_selection["media_selection"])

        # Generate info based in media type
        if selected_type_media_selection["media_selection"] == 'Anime':
            title = anime_info.title.romaji
            logger.info(f"Anime found: {title}\n")
        elif selected_type_media_selection["media_selection"] == 'JDrama':
            print(anime_info)
            # title (movies) -- name (tv)
            title = getattr(anime_info, 'title', anime_info.title)

        # Create folder for saving info.json and segments
        anime_folder_name = map_anime_title_to_media_folder(title)
        anime_folder_fullpath = os.path.join(output_folder, anime_folder_name)
        os.makedirs(anime_folder_fullpath, exist_ok=True)
        logger.info(f"> Base anime folder: {anime_folder_fullpath}")

        info_json_fullpath = os.path.join(anime_folder_fullpath, "info.json")
        logger.info(f"Filepath for info.json: {info_json_fullpath}\n")


        if not os.path.exists(info_json_fullpath):
            logger.info("Creating new info.json file...")
            info_json = {
                "version": "5",
                "folder_media_anime": anime_folder_name
            }

            if selected_type_media_selection["media_selection"] == 'Anime':
                # Completa la info JSON para Anime
                info_json.update({
                    "id": anime_info.id,
                    "type": "anime",
                    "japanese_name": anime_info.title.native,
                    "english_name": anime_info.title.english,
                    "romaji_name": anime_info.title.romaji,
                    "airing_format": anime_info.format,
                    "airing_status": anime_info.status,
                    "genres": anime_info.genres,
                    # "release_date": formatted_date
                })

                cover_url = anime_info.cover.extra_large
                try:
                    banner_url = anime_info.banner
                except AttributeError:
                    banner_url = ""
                    print("WARNING: There is no banner attribute")
                

            elif selected_type_media_selection["media_selection"] == 'JDrama':
                info_json.update({
                    "id": getattr(anime_info, 'id', 'unknown'),
                    "type": "jdrama",
                    "english_name": getattr(anime_info, 'title', getattr(anime_info, 'name', 'unknown')),
                    "airing_format":  getattr(anime_info, "media_type", "unknown"),
                    "airing_status": getattr(anime_info, "status", "unknown"),
                    "genres": [genre.name for genre in getattr(anime_info, 'genres', [])],
                    "release_date": getattr(anime_info, "release_date", getattr(anime_info, "first_air_date", "unknown")).strftime("%Y-%m-%d"),
                })

                cover_url = url_clean(getattr(anime_info, 'poster_path', None))
                banner_url = url_clean(getattr(anime_info, 'backdrop_path', None))

            # Agregar cover y banner al JSON
            if "cover" not in info_json and cover_url:
                cover_data = requests.get(cover_url).content
                cover_filename = f"cover{os.path.splitext(cover_url)[1]}"
                with open(os.path.join(anime_folder_fullpath, cover_filename), "wb") as handler:
                    handler.write(cover_data)
                info_json["cover"] = os.path.join(anime_folder_name, cover_filename)

            if "banner" not in info_json and banner_url:
                banner_data = requests.get(banner_url).content
                banner_filename = f"banner{os.path.splitext(banner_url)[1]}"
                with open(os.path.join(anime_folder_fullpath, banner_filename), "wb") as handler:
                    handler.write(banner_data)
                info_json["banner"] = os.path.join(anime_folder_name, banner_filename)

            # Guardar el archivo JSON
            logger.info(f"Json Data: {info_json}\n")
            with open(info_json_fullpath, "wb") as f:
                # Use utf8 for writing characters correctly
                json_data = json.dumps(info_json, indent=2, ensure_ascii=False).encode("utf8")
                f.write(json_data)

        # Get subtitles
        logger.info("> Finding matching subtitles...")
        matching_subtitles = {}

        # Part 1: Find subtitle files on same directory as episode, with same episode number
        input_episode_parent_folder = Path(episode_filepath).parent
        subtitle_filepaths = [
            os.path.join(input_episode_parent_folder, filename)
            for filename in os.listdir(input_episode_parent_folder)
            if filename.endswith(".ass") or filename.endswith(".srt")
        ]
        logger.debug(f"Subtitle filepaths: {subtitle_filepaths}")

        for subtitle_filepath in subtitle_filepaths:
            subtitle_filename = re.sub(r"\[.*?\]|\(.*?\)", "", os.path.basename(subtitle_filepath))
            if not subtitle_filename.strip():
                logger.error(f"Nombre de archivo de subtítulo inválido: {subtitle_filepath}")
                continue
            guessed_subtitle_info = guessit(subtitle_filename)
            if "episode" in guessed_subtitle_info:
                subtitle_episode = guessed_subtitle_info["episode"]
            else:
                episode_matches = re.search(r"(?!S)(\D\d\d|\D\d)\D", subtitle_filename)
                if episode_matches:
                    subtitle_episode = episode_matches.group(1)
                else:
                    logger.info(
                        "> Could not guess Episode number for subtitle: {subtitle_filepath}"
                    )

            if int(subtitle_episode) == int(episode_info["episode"]):
                logger.info(
                    f"> (E{subtitle_episode}) Found external subtitle: {subtitle_filepath}"
                )

                subtitle_language = None
                if "subtitle_language" in guessed_subtitle_info:
                    subtitle_language = guessed_subtitle_info[
                        "subtitle_language"
                    ].alpha2
                else:
                    try:
                        subtitle_data = pysubs2.load(subtitle_filepath)

                        # Concatenate all the subtitle lines into a single string for better accuracy
                        subtitle_text = " ".join(
                            [event.text for event in subtitle_data]
                        )

                        # Use langdetect to guess the language
                        subtitle_language = detect(subtitle_text)
                        logger.info(
                            f"> External subtitle detected language: {subtitle_language}"
                        )
                    except Exception as e:
                        logger.error(f"Failed to detect language for subtitle: {e}")
                        continue

                if not subtitle_language:
                    logger.error(
                        "Impossible to guess the language of the subtitle. Skipping..."
                    )
                    continue

                if subtitle_language not in SUPPORTED_LANGUAGES:
                    logger.info(
                        f"Language {subtitle_language} is currently not supported. Skipping..."
                    )
                    continue

                subtitle_data = pysubs2.load(subtitle_filepath)
                logger.info(f">Found [{subtitle_language}] subtitles: {subtitle_data}")

                if subtitle_language in matching_subtitles and len(subtitle_data) < len(
                    matching_subtitles[subtitle_language]
                ):
                    logger.info(
                        f"Already found better matching subtitles for this language. Skipping..."
                    )
                    continue

                logger.info(f"Saving subtitles: {subtitle_data}\n")
                matching_subtitles[subtitle_language] = MatchingSubtitle(
                    origin="external",
                    filepath=subtitle_filepath,
                    data=subtitle_data,
                )

        # Part 2: extract srt/ass from mkv (WIP)
        # * Get every subtitle and filter it by using a checkbox select
        # * Extract to /tmp
        # * Add subtitles to matching_subtitles
        tmp_output_folder = os.path.join(anime_folder_fullpath, "tmp")
        os.makedirs(tmp_output_folder, exist_ok=True)
        file_probe = ffmpeg.probe(episode_filepath)

        # Generate the list of available subs
        subtitles_dict = {}
        for stream in file_probe["streams"]:
            if stream["codec_type"] == "subtitle":
                index = stream["index"]
                title = stream.get("tags", {}).get("title")
                language = stream.get("tags", {}).get("language")
                title = title if title else language
                if title and language:
                    subtitles_dict[index] = {"title": title, "language": language}

        subtitle_choices = [
            {"name": f"{details['title']} ({details['language']})", "value": index}
            for index, details in subtitles_dict.items()
        ]
        subtitle_choices.append("none")

        subtitle_questions = [
            inquirer.Checkbox(
                "subtitle_streams",
                message="What subtitles do you want to use?",
                choices=subtitle_choices,
            ),
        ]

        # Check if want to remember this selection for future episodes
        current_subtitles_dict = {
            index: subtitles_dict[index]
            for index in subtitles_dict
            if index in subtitles_dict_remembered
        }

        # If there was a previous selection
        if subtitles_dict_remembered:
            # If the current subtitles dictionary is different from the remembered one
            if current_subtitles_dict != subtitles_dict_remembered:
                logger.info(
                    "Previous subtitles used are different from current episode. Asking for selection again..."
                )
                selected_subtitles = inquirer.prompt(subtitle_questions)
                selected_indices = [
                    subtitle["value"]
                    for subtitle in selected_subtitles["subtitle_streams"]
                ]

                subtitle_remember_question = [
                    inquirer.Confirm(
                        "subtitle_remember",
                        message="Do you want to remember this selection for future episodes?",
                        default=False,
                    )
                ]
                selected_remember_subtitles = inquirer.prompt(
                    subtitle_remember_question
                )
                if selected_remember_subtitles["subtitle_remember"]:
                    subtitles_dict_remembered = {
                        index: subtitles_dict[index] for index in selected_indices
                    }
            else:
                # Previous selection if the current and remembered dictionaries are the same
                selected_indices = [index for index in subtitles_dict_remembered]
        else:
            # If it's the first time or if the remembered selection was cleared, ask for the selection
            selected_subtitles = inquirer.prompt(subtitle_questions)
            selected_indices = [
                subtitle["value"] for subtitle in selected_subtitles["subtitle_streams"]
            ]

            subtitle_remember_question = [
                inquirer.Confirm(
                    "subtitle_remember",
                    message="Do you want to remember this selection for future episodes?",
                    default=False,
                )
            ]
            selected_remember_subtitles = inquirer.prompt(subtitle_remember_question)
            if selected_remember_subtitles["subtitle_remember"]:
                subtitles_dict_remembered = {
                    index: subtitles_dict[index] for index in selected_indices
                }

        subtitle_streams = [
            stream
            for stream in file_probe["streams"]
            if stream["codec_type"] == "subtitle"
            and stream["index"] in selected_indices
        ]

        for subtitle_stream in subtitle_streams:
            index = subtitle_stream["index"]
            codec = subtitle_stream["codec_name"]
            tag_language = subtitle_stream["tags"]["language"]

            # Support for non-ISO 639-3 language tags
            tag_language_normalizer = {"fre": "fra", "ger": "deu"}

            if tag_language_normalizer.get(tag_language):
                tag_language = tag_language_normalizer.get(tag_language)

            subtitle_language = babelfish.Language(tag_language).alpha2
            logger.info(
                f"Found internal subtitle stream. Index: {index}. Codec: {codec}. Language: {subtitle_language}"
            )

            if subtitle_language not in SUPPORTED_LANGUAGES:
                logger.info(
                    f"Language {subtitle_language} is currently not supported. Skipping..."
                )
                continue

            output_sub_tmp_filepath = os.path.join(tmp_output_folder, f"tmp.{codec}")

            subprocess.call(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    episode_filepath,
                    "-map",
                    f"0:{index}",
                    "-c",
                    "copy",
                    output_sub_tmp_filepath,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
            logger.info(f"Exported subtitle to: {output_sub_tmp_filepath}")

            subtitle_data = pysubs2.load(output_sub_tmp_filepath)
            logger.info(f">Found [{subtitle_language}] subtitles: {subtitle_data}")

            if subtitle_language in matching_subtitles:
                logger.info(f"> Already matched subtitles for this language!!")

                if (
                    len(subtitle_data) > len(matching_subtitles[subtitle_language])
                    and matching_subtitles[subtitle_language].origin != "external"
                ):
                    logger.info(
                        ">> Current subtitle internal file is longer than previous selected. Overriding..."
                    )
                else:
                    continue

            logger.info(f"Saving subtitles: {subtitle_data}\n")
            output_sub_final_filepath = os.path.join(
                tmp_output_folder,
                f"{anime_folder_name} {season_number_pretty}{episode_number_pretty}.{subtitle_language}.{codec}",
            )
            subtitle_data.save(output_sub_final_filepath)
            matching_subtitles[subtitle_language] = MatchingSubtitle(
                origin="internal",
                filepath=output_sub_final_filepath,
                data=subtitle_data,
            )

        logger.info(f"Matching subtitles: {matching_subtitles}\n")

        # Having matching JP subtitles is required
        if "ja" not in matching_subtitles:
            raise Exception("Could not find Japanese subtitles. Skipping...")

        # Start segmenting file
        logger.info("Start file segmentation...")

        episode_folder_output_path = os.path.join(
            anime_folder_fullpath, season_number_pretty, episode_number_pretty
        )
        os.makedirs(episode_folder_output_path, exist_ok=True)

        if args.parallel:
            pool.apply_async(
                split_video_by_subtitles,
                (
                    translator,
                    episode_filepath,
                    matching_subtitles,
                    episode_folder_output_path,
                    args,
                ),
            )
        else:
            split_video_by_subtitles(
                translator,
                episode_filepath,
                matching_subtitles,
                episode_folder_output_path,
                args,
            )

        # shutil.rmtree(tmp_output_folder, ignore_errors=True)
        logger.info(f"Finished")

    except Exception:
        logger.error(
            "Something happened processing the anime. Skipping...", exc_info=True
        )

    return pool, subtitles_dict_remembered


def split_video_by_subtitles(
    translator,
    video_file,
    subtitles,
    episode_folder_output_path,
    args,
    output_tsv_name="data.tsv",
):
    video = mp.VideoFileClip(video_file) if video_file else None

    # # TODO: Sync subtitles calling ffsubsync
    # Use first found internal sub as reference for timing since it should be 100% perfect

    # > From here on just assume all subtitles are perfectly synced
    synced_subtitles = subtitles

    # Extract all subtitles lines from all subtitle files passed
    sorted_lines = []
    for language, subs in synced_subtitles.items():
        for line in subs.data:
            sentence = process_subtitle_line(line, args)
            sorted_lines.append(
                {
                    "start": line.start,
                    "end": line.end,
                    "language": language,
                    "sentence": sentence,
                    "actor": line.name,
                }
            )

    # Sort all subtitle lines by start timestamp
    sorted_lines.sort(key=lambda x: x["start"])

    # Give an id to each line
    for i, line in enumerate(sorted_lines):
        line["sub_id"] = i
        sorted_lines[i] = line

    # Remove empty lines
    sorted_lines = list(filter(lambda x: x["sentence"], sorted_lines))

    # Remove duplicate lines (with same start, end, sentence and language)
    duplicates_set = set()
    for line in list(sorted_lines):
        # Ignore the attribute `sub_id` so we can detect duplicates
        line_hashkey = (line["start"], line["end"], line["language"], line["sentence"])

        if line_hashkey not in duplicates_set:
            duplicates_set.add(line_hashkey)
        else:
            sorted_lines.remove(line)

    tsv_filepath = os.path.join(episode_folder_output_path, output_tsv_name)
    with open(tsv_filepath, "w+", newline="", encoding="utf-8") as tsvfile:
        writer = csv.DictWriter(
            tsvfile,
            fieldnames=EpisodeTsvRow._fields,
            delimiter="\t",
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
        )
        writer.writeheader()

        segment_start = sorted_lines[0]["start"] - 1
        segment_end = sorted_lines[0]["end"] + 1
        segment_sentences = {}
        line_logs = [episode_folder_output_path, ""]
        for i, line in enumerate(sorted_lines):
            ln = line["language"]

            # New line when:
            #   * No overlap
            #   * Overlap, but gap is smaller than 500
            if not (segment_start < line["end"] and line["start"] < segment_end) or (
                (segment_start < line["end"] and line["start"] < segment_end)
                and abs(segment_end - line["start"]) < 1100
            ):
                if "ja" in segment_sentences and (
                    "en" in segment_sentences or "es" in segment_sentences
                ):
                    segment_logs = generate_segment(
                        i,
                        segment_sentences,
                        segment_start,
                        segment_end,
                        episode_folder_output_path,
                        video,
                        translator,
                        writer,
                        args,
                    )
                    if segment_logs:
                        line_logs = line_logs + segment_logs

                else:
                    line_logs.append("No en/es subtitle match. Ignoring...\n")

                line_logs.append("-------------------------------------------------")
                logger.info("\n".join(line_logs))
                line_logs = [episode_folder_output_path, ""]
                line_logs.append(f"[{ln}] Line: {line}")

                segment_sentences = {ln: [line]}
                segment_start = line["start"]
                segment_end = line["end"]

            else:
                line_logs.append(f"[{ln}] Line: {line}")
                segment_sentences[ln] = segment_sentences.get(ln, [])

                # Sometimes when two characters are speaking the same line is repeated several times. Detect that
                # to avoid duplicating the same sentence
                eq_match = False
                for saved_line in segment_sentences[ln]:
                    if (
                        saved_line["sentence"] == line["sentence"]
                        and segment_sentences[ln][-1]["end"] == line["start"]
                    ):
                        eq_match = True

                if not eq_match:
                    segment_sentences[ln].append(line)

                segment_start = min(segment_start, line["start"])
                segment_end = max(segment_end, line["end"])


def generate_segment(
    i,
    segment_sentences,
    segment_start,
    segment_end,
    output_path,
    video,
    translator,
    writer,
    args,
):
    logs = []
    sentence_japanese, actor_japanese, subs_jp_ids = join_sentences_to_segment(
        segment_sentences["ja"], "ja"
    )
    sentence_english, actor_english, subs_en_ids = (
        join_sentences_to_segment(segment_sentences["en"], "en")
        if "en" in segment_sentences
        else (None, None, [])
    )
    sentence_spanish, actor_spanish, subs_es_ids = (
        join_sentences_to_segment(segment_sentences["es"], "es")
        if "es" in segment_sentences
        else (None, None, [])
    )
    # Use ID of the japanese sentence to identify the whole segment, since we always
    # have to include japanese subtitles
    segment_id = subs_jp_ids[0]

    sentence_spanish_is_mt = False if sentence_spanish else None
    sentence_english_is_mt = False if sentence_english else None

    if translator and not sentence_spanish:
        sentence_spanish = translator.translate_text(
            sentence_japanese, source_lang="JA", target_lang="ES"
        ).text
        sentence_spanish_is_mt = True
        logs.append(f"[DEEPL - SPANISH]: {sentence_spanish}")

    if translator and not sentence_english:
        sentence_english = translator.translate_text(
            sentence_japanese, source_lang="JA", target_lang="EN-US"
        ).text
        sentence_english_is_mt = True
        logs.append(f"[DEEPL - ENGLISH]: {sentence_english}")

    start_time_delta = timedelta(milliseconds=segment_start)
    start_time_seconds = start_time_delta.total_seconds()
    end_time_delta = timedelta(milliseconds=segment_end)
    end_time_seconds = end_time_delta.total_seconds()

    subs_jp_ids_str = ",".join(list(map(str, subs_jp_ids)))
    subs_es_ids_str = ",".join(list(map(str, subs_es_ids)))
    subs_en_ids_str = ",".join(list(map(str, subs_en_ids)))
    logs.append(f"({segment_id}) {start_time_delta} - {end_time_delta}")
    logs.append(f"[JA] ({subs_jp_ids_str}) {sentence_japanese}")
    logs.append(f"[ES] ({subs_es_ids_str}) {sentence_spanish}")
    logs.append(f"[EN] ({subs_en_ids_str}) {sentence_english}")

    audio_filename = f"{segment_id}.mp3"
    screenshot_filename = f"{segment_id}.webp"
    video_filename = f"{segment_id}.mp4"

    # Audio
    if video and not args.dryrun:
        try:
            subclip = video.subclip(start_time_seconds, end_time_seconds)
            audio = subclip.audio
            audio_path = os.path.join(output_path, audio_filename)

            audio.write_audiofile(audio_path, codec="mp3", logger=None)

            logs.append(f"> Saved audio in {audio_path}")

        except Exception as err:
            logger.exception(f"Error creating audio '{audio_filename}'", err)
            return

        # Screenshot
        try:
            screenshot_path = os.path.join(output_path, screenshot_filename)

            # Take a screenshot on the middle of the dialog
            screenshot_time = (start_time_seconds + end_time_seconds) / 2
            video.save_frame(screenshot_path, t=screenshot_time)

            logs.append(f"> Saved screenshot in {screenshot_path}")

        except Exception as err:
            logger.exception(f"Error creating screenshot '{screenshot_filename}'", err)
            return

        # Video
        video_path = os.path.join(output_path, video_filename)
        video_length_delta = end_time_delta - start_time_delta

        try:
            subprocess.call(
                [
                    "ffmpeg",
                    "-y",
                    "-loop",
                    "1",
                    "-framerate",
                    "10",
                    "-i",
                    screenshot_path,
                    "-i",
                    audio_path,
                    "-vf",
                    "scale=1280:720,setsar=1",
                    "-c:v",
                    "libx264",
                    "-tune",
                    "stillimage",
                    "-b:v",
                    "200k",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    "-t",
                    str(video_length_delta),
                    video_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT
            )
            logs.append(f"> Saved video in {video_path}")

        except subprocess.CalledProcessError as err:
            logger.exception(f"Error creating video `{video_path}", err)
            return

    writer.writerow(
        EpisodeTsvRow(
            ID=segment_id,
            SUBS_JP_IDS=subs_jp_ids_str,
            SUBS_ES_IDS=subs_es_ids_str,
            SUBS_EN_IDS=subs_en_ids_str,
            START_TIME=start_time_delta,
            END_TIME=end_time_delta,
            NAME_AUDIO=audio_filename,
            NAME_SCREENSHOT=screenshot_filename,
            CONTENT=sentence_japanese,
            CONTENT_TRANSLATION_SPANISH=sentence_spanish,
            CONTENT_TRANSLATION_ENGLISH=sentence_english,
            CONTENT_SPANISH_MT=sentence_spanish_is_mt,
            CONTENT_ENGLISH_MT=sentence_english_is_mt,
            ACTOR_JA=actor_japanese,
            ACTOR_ES=actor_spanish,
            ACTOR_EN=actor_english,
        )._asdict()
    )
    logs.append("Segment saved!\n")
    return logs


def join_sentences_to_segment(sentences, ln):
    join_symbol = "　" if ln == "ja" else " "
    joined_sentence = join_symbol.join(map(lambda x: x["sentence"].strip(), sentences))

    # Sometimes japanese subs don't use the appropriate " symbol for quotes
    invalid_quotes = r"``|''"
    joined_sentence = re.sub(invalid_quotes, '"', joined_sentence)

    # On certain cases it makes sense to not add a - since there is another symbol
    # Already indicating the end of the sentence
    remove_redundant_symbols = [
        r"(?<=\.\.\.)-",
        r"(?<=\?)-",
        r"(?<=!)-",
        r"(?<=\.)-",
        r"(?<=,)-",
        r"(?<=ー)-",
        r"(?<=-)-",
        r"(?<=。)\s",
        r"^-",
        r"(?<=\s)+\s",
        r"(?<=\.\.\.)。",
    ]

    actor_sentence = ",".join(
        sorted(set(map(lambda x: x["actor"].replace("\t", "").strip(), sentences)))
    )

    # Get all the ids that form the segment
    subs_ids = list(map(lambda s: s["sub_id"], sentences))

    return (
        re.sub(rf"{'|'.join(remove_redundant_symbols)}", "", joined_sentence),
        actor_sentence,
        subs_ids,
    )


def process_subtitle_line(line, args):
    if line.type != "Dialogue":
        return ""

    # Ass subtitles include an actor name that sometimes can be used to filter
    # non-dialog subtitles
    if line.name and re.search(r"sign|[_\-\s]?ed|op[_\-\s]?", line.name.lower()):
        return ""

    # *Top, sign... is usually used for background conversations with an ongoing
    # dialog

    if line.style and re.search(r"top|sign|tipo tv|block|alt|cart", line.style.lower()):
        return ""

    # Sometimes .ass subtitles include the signs subs on the main dialog
    # Skip all lines that have pos() or move() ass method as it is not a real dialog line
    if re.search(r"pos\(.*?\)|move\(.*?\)|♬", line.text):
        return ""

    # Normaliza half-width (Hankaku) a full-width (Zenkaku) caracteres
    processed_sentence = jaconvV2.normalize(line.plaintext, "NFKC")

    # Replace all new lines / tabs / separators with just one space
    processed_sentence = re.sub("\r?\n|\t", " ", processed_sentence)

    if hasattr(args, "extra_punctuation") and args.extra_punctuation:
        processed_sentence = processed_sentence.replace("・", " ")

    processed_sentence = remove_nested_parenthesis(processed_sentence)

    special_chars = r"⚟|⚞|<|>|=|●|→|ー?♪ー?|\u202a|\u202c|➡|&lrm;"
    processed_sentence = re.sub(special_chars, "", processed_sentence)

    processed_sentence = emoji.sub("", processed_sentence)

    return processed_sentence.strip()


def remove_nested_parenthesis(sentence):
    nb_rep = 1
    while nb_rep:
        (sentence, nb_rep) = re.subn(
            r"\([^\(\)（）\[\]\{\}《》【】]*\)|\[[^\(\)（）\[\]\{\}《》【】]*\]", "", sentence
        )

    return sentence


def extract_anime_title_for_guessit(episode_filepath):
    """
    This method tries to parse the full episode path and get a coherent anime title. This methods does the following
    postprocessing:
      * Take only the episode name and the parent folder name
      * Remove everything between [ and ]. This is usually the encoder name or the file ID
      * Remove tags related to file quality and format (1080p/720p, Audio, HEVC, x265, BDRip...)

    Example:
      * Input:  Shingeki No Kyojin S01 1080p BDRip 10 bits x265-EMBER/S01E01- To You, in 2000 Years [14197707]
      * Output: Shingeki No Kyojin S01 -EMBER S01E01- To You, in 2000 Years

    This allows guessit to return "Shingeki No Kyojin" as the anime title, instead of returning the episode title
    """
    return re.sub(
        r"\[.*?\]|1080p|720p|BDRip|Dual\s?Audio|x?26[4|5]-?|HEVC|10\sbits|EMBER",
        "",
        " ".join(episode_filepath.split("/")[-2:]),
    )


def extract_anime_title_for_anilist(guessed_anime_title):
    """
    After extracting the name from Guessit, we have to do a bit more of postprocessing because Anilist is really
    sensitive with the title search. Including extra information like season or episodoe number will case Anilist
    to return nothing:
        * Remove Season and Episode numbers
    """
    return re.sub(r"S\d.*?(\s|$)", "", guessed_anime_title).strip()


def map_anime_title_to_media_folder(anime_title):
    """
    Root folder for all the anime information (subfolders for seasons/episodes, info.json, etc) will be stored using
    lower case, kebab case, without any punctuation or invalid symbols

    Example:
        * Input: Mobile Suit Gundam: The Witch from Mercury
        * Ooutput: mobile-suit-gundam-the-witch-from-mercury
    """
    return "-".join(
        anime_title.lower().translate(str.maketrans("", "", string.punctuation)).split()
    )


class CachedMediaInfo:
    def __init__(self):
        self.client_anilist = Client()
        self.client_imdb = Cinemagoer()
        self.cached_results = {}
        self.tmdb = TMDb(key='d31986f7c2be74a9685649eb917b9e25', language="en-US", region="US")

    def get_media_info(self, search_query, content_type):
        if search_query in self.cached_results:
            return self.cached_results[search_query]

        if content_type == "Anime":
            search_results = self.client_anilist.search(search_query)
            logger.debug("Search results for anime", search_results)

            if not search_results:
                raise Exception(
                    f"Anime with title {search_results} not found. Please check file name"
                )

        elif(content_type == "JDrama"):
            search_results = self.tmdb.search().multi(search_query)

            if not search_results:
                raise Exception(
                    f"JDrama with title {search_results} not found. Please check file name"
                )

        selected_index = 0
        if len(search_results) > 1:
            logger.info("Multiple results found! Please select the better match")
            for i, result in enumerate(search_results):
                if content_type == "Anime":
                    title = result['title'] if content_type == "JDrama" else result.title.romaji
                    logger.info(f"[{i}]: {title}")
                elif content_type == "JDrama":
                    if result.media_type == 'tv':
                        title = result.name
                        first_air_date = result.first_air_date
                        print(f"[{i}]: {title}, First air date: {first_air_date}, {result.media_type}")
                    elif result.media_type == 'movie':
                        title = result.title
                        release_date = result.release_date
                        print(f"[{i}]: {title}, Release date: {release_date}, {result.media_type}")

            selected_index = int(input("> Please select a number: "))

        selected_result = search_results[selected_index]
                        
        if content_type == "Anime":
            detailed_result = self.client_anilist.get_anime(selected_result.id)
        elif content_type == "JDrama":
            if(selected_result.media_type == "movie"):
                detailed_result = self.tmdb.movie(selected_result.id).details(append_to_response="external_ids,images,videos")
            elif(selected_result.media_type == "tv"):
                detailed_result = self.tmdb.tv(selected_result.id).details(append_to_response="external_ids,images,videos")

        self.cached_results[search_query] = detailed_result

        return detailed_result


def command_args():
    parser = argparse.ArgumentParser(
        description="Split one or several .mkv files onto separate audio segments with images"
    )
    parser.add_argument(
        "input", type=pathlib.Path, help="Input folder with .mkv files and subtitles"
    )
    parser.add_argument(
        "output",
        type=pathlib.Path,
        help="Output folder",
    )
    parser.add_argument(
        "-t",
        "--token",
        dest="token",
        type=str,
        help="DeepL token for translating subtitles. If not provided, the only generated subtitles will be taken from "
        "existing subtitle files",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add extra debug information to the execution",
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        dest="dryrun",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Execute and parse subtitles, but without generating the segments",
    )
    parser.add_argument(
        "-x",
        "--x",
        dest="extra_punctuation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Remove other common punctuation symbols like ・. This might cause certain"
        "subtitles to lose fidelity.",
    )
    parser.add_argument(
        "-p",
        "--parallel",
        dest="parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate segments for episodes in parallel",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
