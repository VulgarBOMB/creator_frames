import os
import shutil
import fitparse
import datetime, time
import pandas as pd
import numpy as np
# ----------------------
from scipy import interpolate
from math import radians, cos, sin, sqrt, atan2
from PIL import Image, ImageDraw, ImageFont, ImageChops
from widgets_config_1920_1080 import FONT_PATH, COLORS, WIDGET_CONFIG

# ----------------------
# КОНФИГУРАЦИЯ
# ----------------------
INPUT_FIT_FILE = "ride-0-2025-06-10-13-37-27.fit"
OUTPUT_FOLDER = "telemetry_frames"
FPS = 16
RESOLUTION = (1920, 1080)

# Временной диапазон для обрезки
TRIM_START = "2025/06/10 10:37:34"
TRIM_END = "2025/06/10 11:37:34"

# Глобальные переменные
cached_background = None
cached_map_bounds = None
SPEED_HISTORY = []

# Удаление старой директории
shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ----------------------
# 1. ПАРСИНГ FIT-ФАЙЛА
# ----------------------
# Функция для вычисления расстояния между двумя точками в градусах
def fit_to_dataframe(fit_file, user_input=None):  # Добавлен параметр для координат
    fitfile = fitparse.FitFile(fit_file)
    records = []

    for record in fitfile.get_messages('record'):
        r = {}
        for data in record:
            if data.value is not None:
                if data.name == 'enhanced_altitude':
                    r['altitude'] = data.value
                elif data.name == 'altitude':
                    r['altitude'] = data.value
                elif data.name == 'speed':
                    r[data.name] = data.value * 3.6
                else:
                    r[data.name] = data.value

        required_fields = ['timestamp', 'speed', 'heart_rate', 'power', 'distance', 'latitude', 'longitude']
        for field in required_fields:
            if field not in r:
                r[field] = None

        if 'position_lat' in r and 'position_long' in r:
            r['latitude'] = r['position_lat'] * (180 / 2 ** 31)
            r['longitude'] = r['position_long'] * (180 / 2 ** 31)

        records.append(r)

    df = pd.DataFrame(records).dropna(subset=['timestamp'])

    # Преобразуем timestamp в datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Проверка временного диапазона
    min_ts = df['timestamp'].min()
    max_ts = df['timestamp'].max()

    trim_start = pd.to_datetime(TRIM_START)
    trim_end = pd.to_datetime(TRIM_END)

    if not (min_ts <= trim_start <= max_ts):
        print(f"Ошибка: TRIM_START ({trim_start}) выходит за пределы доступных временных меток")
        print(f"Минимальное значение timestamp: {df['timestamp'].min()}")
        print(f"Максимальное значение timestamp: {df['timestamp'].max()}")
        exit(1)

    if not (min_ts <= trim_end <= max_ts):
        print(f"Ошибка: TRIM_END ({trim_end}) выходит за пределы доступных временных меток")
        print(f"Минимальное значение timestamp: {df['timestamp'].min()}")
        print(f"Максимальное значение timestamp: {df['timestamp'].max()}")
        exit(1)

    if TRIM_START and TRIM_END:
        # mask = (df['Time'] >= pd.to_datetime(TRIM_START)) & (df['Time'] <= pd.to_datetime(TRIM_END))
        mask = (df['timestamp'] >= pd.to_datetime(TRIM_START)) & (df['timestamp'] <= pd.to_datetime(TRIM_END))
        df = df.loc[mask]

    # # Новая логика для поиска ближайшей точки
    # user_input = input("Введите координаты в формате 'latitude, longitude': ")
    # if user_input:
    #     try:
    #         user_lat, user_lon = map(float, user_input.split(','))
    #     except:
    #         print("Ошибка: неверный формат ввода координат")
    #         exit(1)
    #
    #     # Функция расчета расстояния между точками
    #     def haversine(lat1, lon1, lat2, lon2):
    #         R = 6371  # радиус Земли в км
    #         phi1 = radians(lat1)
    #         phi2 = radians(lat2)
    #         delta_phi = radians(lat2 - lat1)
    #         delta_lambda = radians(lon2 - lon1)
    #         a = sin(delta_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2) ** 2
    #         c = 2 * atan2(sqrt(a), sqrt(1 - a))
    #         return R * c
    #
    #     # Вычисляем расстояние до введенных координат для каждой строки
    #     df['distance'] = df.apply(
    #         lambda row: haversine(row['latitude'], row['longitude'], user_lat, user_lon),
    #         axis=1
    #     )
    #
    #     # Находим строку с минимальным расстоянием
    #     closest_row = df.loc[df['distance'].idxmin()]
    #     print(f"Ближайшая точка найдена: {closest_row['timestamp'].strftime('%Y/%m/%d %H:%M:%S')}")

    return df

# ----------------------
# 2. ФУНКЦИЯ ПОСТРОЕНИЯ ГРАФИКА ПО ДИСТАНЦИИ
# ----------------------
def create_static_altitude_graph(df, config, min_dist, max_dist, min_alt, max_alt):
    width, height = config['size']
    padding = config.get('padding', {'left': 0, 'right': 0, 'top': 0, 'bottom': 0})
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    if df.empty or df['altitude'].isna().all():
        return img

    df = df.dropna(subset=['distance', 'altitude'])
    if len(df) < 2:
        return img

    # Основные параметры графика
    dist_range = max_dist - min_dist if max_dist != min_dist else 1
    alt_range = max_alt - min_alt if max_alt != min_alt else 1

    plot_width = width - padding['left'] - padding['right']
    plot_height = height - padding['top'] - padding['bottom']

    # Векторизованный расчёт координат
    distances = df['distance'].values
    altitudes = df['altitude'].values

    x_coords = padding['left'] + ((distances - min_dist) / dist_range * plot_width).astype(int)
    y_coords = padding['top'] + plot_height - ((altitudes - min_alt) / alt_range * plot_height).astype(int)

    points = list(zip(x_coords, y_coords))

    # --- Добавляем заливку под графиком ---
    if len(points) > 1:
        # Создаём полигон: точки графика + нижние углы
        polygon_points = points.copy()
        polygon_points.append((points[-1][0], height - padding['bottom']))  # правый нижний угол
        polygon_points.append((points[0][0], height - padding['bottom']))   # левый нижний угол

        # Рисуем заливку под графиком
        draw.polygon(polygon_points, fill=COLORS['graph_fill'])

    # Рисуем сам график
    if len(points) >= 2:
        draw.line(points, fill=config['line_color'], width=config['line_width'])

    return img

def create_altitude_point(static_graph_img, current_row, config, min_dist, max_dist, min_alt, max_alt):
    current_row = current_row._asdict()
    width, height = config['size']
    padding = config.get('padding', {'left': 0, 'right': 0, 'top': 0, 'bottom': 0})
    img = static_graph_img.copy()  # Копируем статичный график
    draw = ImageDraw.Draw(img)

    dist_range = max_dist - min_dist if max_dist != min_dist else 1
    alt_range = max_alt - min_alt if max_alt != min_alt else 1

    plot_width = width - padding['left'] - padding['right']
    plot_height = height - padding['top'] - padding['bottom']

    try:
        current_x = padding['left'] + int((current_row['distance'] - min_dist) / dist_range * plot_width)
        current_y = padding['top'] + plot_height - int((current_row['altitude'] - min_alt) / alt_range * plot_height)

        point_size = config.get('point_size', 8)
        draw.ellipse(
            (current_x - point_size * 2, current_y - point_size * 2,
             current_x + point_size * 2, current_y + point_size * 2),
            fill=COLORS['current_point']
        )

        # Текст с высотой
        alt_text = f"{current_row['altitude']:.0f}m"
        font = ImageFont.truetype(FONT_PATH, config['font_size'])

        text_bbox = draw.textbbox((0, 0), alt_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        base_x = current_x
        base_y = current_y
        text_position = config.get('text_position', 'right')

        if text_position == 'right':
            base_x += point_size * 2
            base_y -= text_height // 2
        elif text_position == 'left':
            base_x -= point_size * 2 + text_width
            base_y -= text_height // 2
        elif text_position == 'top':
            base_x -= text_width // 2
            base_y -= point_size * 2 + text_height
        elif text_position == 'bottom':
            base_x -= text_width // 2
            base_y += point_size * 2

        text_x = base_x + config.get('text_offset', {}).get('horizontal', 0)
        text_y = base_y + config.get('text_offset', {}).get('vertical', 0)

        draw.text((text_x, text_y), alt_text, font=font, fill=COLORS['text'])

    except Exception as e:
        print(f"Ошибка при отрисовке точки: {e}")

    return img

# ----------------------
# 3. ФУНКЦИЯ ОТРИСОВКИ ТЕКСТОВЫХ ВИДЖЕТОВ
# ----------------------
def create_text_widget(draw, text, config):
    font = ImageFont.truetype(FONT_PATH, config['font_size'])
    full_text = text
    pos_x, pos_y = config['position']
    draw.text((pos_x, pos_y), full_text, font=font, fill=COLORS['text'])

# ----------------------
# 4. ФУНКЦИЯ ОТРИСОВКИ МИНИ-КАРТЫ
# ----------------------
def load_map_background(config):
    global cached_background, cached_map_bounds

    map_image_path = config.get('map_image', 'map_background.png')
    map_bounds = config.get('map_bounds', {})

    # Если уже загружено и границы совпадают — возвращаем кэш
    if cached_background is not None and cached_map_bounds == map_bounds:
        return cached_background, map_bounds

    # Иначе загружаем заново
    try:
        background = Image.open(map_image_path).convert("RGBA")
        bg_width, bg_height = background.size
        cached_background = background
        cached_map_bounds = map_bounds
        print(f"[{datetime.datetime.now()}] Фон карты загружен")
        return background, map_bounds
    except FileNotFoundError:
        print(f"Ошибка: Файл {map_image_path} не найден")
        return Image.new("RGBA", config['size'], (50, 50, 50, 255)), map_bounds

def create_minimap(window_df, current_row, config, background=None, map_bounds=None):
    def gps_to_pixel(lon, lat, map_bounds, image_size):
        img_width, img_height = image_size
        min_lon, max_lon = map_bounds['min_lon'], map_bounds['max_lon']
        min_lat, max_lat = map_bounds['min_lat'], map_bounds['max_lat']
        x = (lon - min_lon) / (max_lon - min_lon) * img_width
        y = (max_lat - lat) / (max_lat - min_lat) * img_height
        return (int(x), int(y))

    # Параметры из конфига
    width, height = config['size']
    point_size = config.get('point_size', 4)

    # Используем кэшированное изображение
    if background is None or map_bounds is None:
        background, map_bounds = load_map_background(config)
        if isinstance(background, Image.Image):
            bg_width, bg_height = background.size
        else:
            return Image.new("RGBA", (width, height), (0, 0, 0, 0))
    else:
        bg_width, bg_height = background.size

    # Получаем текущие координаты
    if isinstance(current_row, dict):
        current_lat = current_row.get('latitude')
        current_lon = current_row.get('longitude')
    else:
        current_lat = getattr(current_row, 'latitude', None)
        current_lon = getattr(current_row, 'longitude', None)

    if None in (current_lat, current_lon):
        return Image.new("RGBA", (width, height), (0, 0, 0, 0))

    # Переводим GPS в пиксельные координаты
    try:
        center_x, center_y = gps_to_pixel(current_lon, current_lat, map_bounds, (bg_width, bg_height))
    except ZeroDivisionError:
        return Image.new("RGBA", (width, height), (0, 0, 0, 0))

    # Скользящее среднее по скорости
    current_speed = current_row.get('speed', 0) if isinstance(current_row, dict) else getattr(current_row, 'speed', 0)
    global SPEED_HISTORY
    SPEED_HISTORY.append(current_speed)
    if len(SPEED_HISTORY) > config.get('window_size', 40):
        SPEED_HISTORY.pop(0)
    smoothed_speed = sum(SPEED_HISTORY) / len(SPEED_HISTORY)

    # Расчёт масштаба
    base_speed = config.get('base_speed', 10)
    scale_factor = max(1, min(base_speed / max(1, smoothed_speed), 3))

    # Размер обрезаемого участка
    new_cropped_width = int(width / scale_factor)
    new_cropped_height = int(height / scale_factor)

    # Коррекция позиции
    half_width = new_cropped_width // 2
    half_height = new_cropped_height // 2

    left = max(0, min(bg_width - new_cropped_width, center_x - half_width))
    top = max(0, min(bg_height - new_cropped_height, center_y - half_height))
    right = left + new_cropped_width
    bottom = top + new_cropped_height

    # Обрезка и масштабирование
    cropped_map = background.crop((left, top, right, bottom))
    resized_map = cropped_map.resize((width, height), Image.Resampling.LANCZOS)

    # Базовое изображение мини-карты
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    img.paste(resized_map, (0, 0))

    # Фильтруем видимые точки
    visible_path = window_df[
        (window_df['longitude'] >= map_bounds['min_lon']) &
        (window_df['longitude'] <= map_bounds['max_lon']) &
        (window_df['latitude'] >= map_bounds['min_lat']) &
        (window_df['latitude'] <= map_bounds['max_lat'])
    ]

    if len(visible_path) < 2:
        return img

    # Переводим все точки в пиксельные координаты
    points = []
    for _, r in visible_path.iterrows():
        x, y = gps_to_pixel(r['longitude'], r['latitude'], map_bounds, (bg_width, bg_height))
        x -= left
        y -= top
        points.append((x, y))

    # Масштабируем точки
    scaled_points = [(int(x * scale_factor), int(y * scale_factor)) for x, y in points]

    # Рисуем текущую точку
    draw.ellipse(
        (width // 2 - point_size * 2, height // 2 - point_size * 2,
         width // 2 + point_size * 2, height // 2 + point_size * 2),
        fill=COLORS['current_point']
    )

    # Создаём маску с белой обводкой
    mask = Image.new('L', (width, height), 0)
    draw_mask = ImageDraw.Draw(mask)
    draw_mask.ellipse((0, 0, width - 1, height - 1), fill=255)

    # Обводка
    border_size = config.get('line_width', 2)
    border_mask = Image.new('L', (width + border_size * 2, height + border_size * 2), 0)
    border_draw = ImageDraw.Draw(border_mask)
    border_draw.ellipse((border_size, border_size, width + border_size - 1, height + border_size - 1), outline=255, width=border_size * 2)
    border_mask = border_mask.crop((border_size, border_size, border_size + width, border_size + height))

    # Комбинируем маску и обводку
    final_mask = ImageChops.add(mask, border_mask)

    # Применяем маску
    final_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    final_img.paste(img, (0, 0), mask=final_mask)

    # Рисуем обводку
    draw_final = ImageDraw.Draw(final_img)
    draw_final.ellipse(
        (border_size // 2, border_size // 2,
         width - border_size // 2, height - border_size // 2),
        outline='white',
        width=border_size
    )

    return final_img

# ----------------------
# 5. ФУНКЦИЯ ОТРИСОВКИ ПРОГРЕСС-БАРА
# ----------------------
def create_progress_bar(draw, current_value, config, widget_key):
    # Вычисляем динамические min/max из данных
    if 'heart_rate' in widget_key:
        min_val = interp_df['heart_rate'].min()
        max_val = interp_df['heart_rate'].max()
    else:  # power_progress
        min_val = interp_df['power'].min()
        max_val = interp_df['power'].max()

    x, y = config['position']
    bar_width, bar_height = config['size']
    border_color = config['border_color']
    fill_color = config['fill_color']
    text_color = config.get('text_color', (255, 255, 255))
    font_size = config.get('font_size', bar_height)
    line_width = config.get('line_width', 2)
    font = ImageFont.truetype(FONT_PATH, font_size)

    # Основная рамка
    draw.rectangle(
        [(x, y), (x + bar_width, y + bar_height)],
        outline=border_color,
        width=line_width
    )

    # Расчет заполнения
    if max_val == min_val:
        fill_ratio = 0.0 if current_value <= min_val else 1.0
    else:
        fill_ratio = (current_value - min_val) / (max_val - min_val)
    fill_width = bar_width * fill_ratio

    # Заполнение
    draw.rectangle(
        [
            (x + line_width // 2, y + line_width // 2),
            (x + line_width // 2 + fill_width, y + bar_height - line_width // 2)
        ],
        fill=fill_color
    )

    # Текстовые метки
    text_offset_y = -font_size - 5  # Позиция над баром

    # Минимальное значение слева (центр текста на левом краю бара)
    min_text = f"{int(min_val)}"
    min_text_bbox = draw.textbbox((0, 0), min_text, font=font)
    min_text_width = min_text_bbox[2] - min_text_bbox[0]
    min_text_x = x - min_text_width // 2  # Центр текста = левый край бара
    draw.text((min_text_x, y + text_offset_y), min_text, fill=text_color, font=font)

    # Максимальное значение справа (центр текста на правом краю бара)
    max_text = f"{int(max_val)}"
    max_text_bbox = draw.textbbox((0, 0), max_text, font=font)
    max_text_width = max_text_bbox[2] - max_text_bbox[0]
    max_text_x = x + bar_width - (max_text_width // 2)  # Центр текста = правый край бара
    draw.text((max_text_x, y + text_offset_y), max_text, fill=text_color, font=font)

    # Текущее значение в центре (с постфиксом из конфига)
    postfix = config.get('postfix', '')
    current_text = f"{int(current_value)}{postfix}"
    current_text_bbox = draw.textbbox((0, 0), current_text, font=font)
    current_text_width = current_text_bbox[2] - current_text_bbox[0]
    current_text_x = x + (bar_width // 2) - (current_text_width // 2)
    draw.text((current_text_x, y + text_offset_y), current_text, fill=text_color, font=font)

# ----------------------
# 6. ПОДГОТОВКА ДАННЫХ
# ----------------------
print("Processing FIT file...")
time_of_script_start = datetime.datetime.now()
df = fit_to_dataframe(INPUT_FIT_FILE)
df = df.sort_values('timestamp').drop_duplicates('timestamp')
df.to_csv('output.csv')

start_time = df['timestamp'].iloc[0]
df['time_sec'] = (df['timestamp'] - start_time).dt.total_seconds()
valid_columns = ['altitude', 'speed', 'heart_rate', 'power', 'distance', 'latitude', 'longitude', 'grade', 'timestamp']
df['grade'] = (df['altitude'].diff() / df['distance'].diff()).replace([np.inf, -np.inf], np.nan).fillna(0) * 100
df = df.dropna(subset=valid_columns).reset_index(drop=True)

# Интерполяция данных
new_time_sec = np.arange(
    start=df['time_sec'].min(),
    stop=df['time_sec'].max(),
    step=1 / FPS
)

interp_funcs = {
    col: interpolate.interp1d(
        df['time_sec'],
        df[col],
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate'
    ) for col in valid_columns
}

interp_df = pd.DataFrame({
    'time_sec': new_time_sec,
    **{col: interp_funcs[col](new_time_sec) for col in valid_columns}
})

interp_df.to_csv('output.csv')

# Вычисляем минимумы и максимумы
heart_rate_min = interp_df['heart_rate'].min()
heart_rate_max = interp_df['heart_rate'].max()
power_min = interp_df['power'].min()
power_max = interp_df['power'].max()

# Обработка NaN
if pd.isna(heart_rate_min):
    heart_rate_min = 0
if pd.isna(power_min):
    power_min = 0

# ----------------------
# 7. ГЕНЕРАЦИЯ КАДРОВ
# ----------------------

# Инициализация статистики
frame_times = []
altitude_times = []
minimap_times = []
widget_times = []
progress_bar_times = []
last_log_time = time.time()
total_frames = len(interp_df)

# Инициализация глобальных метрик
min_dist = interp_df['distance'].min()
max_dist = interp_df['distance'].max()
min_alt = interp_df['altitude'].min()
max_alt = interp_df['altitude'].max()

# Создаем статичный график один раз
static_altitude_img = create_static_altitude_graph(interp_df, WIDGET_CONFIG['altitude_graph'], min_dist, max_dist, min_alt, max_alt)
static_altitude_img.save(r'D:\PythonProject\auto_ru_parser\back.png')

# Фильтруем только текстовые виджеты
text_widgets = [key for key in list(WIDGET_CONFIG.keys()) if key in ('speed',
                                                                     'KMH',
                                                                     'grade')]

# Загружаем фон карты один раз
minimap_background, minimap_bounds = load_map_background(WIDGET_CONFIG['minimap'])

for idx, row in enumerate(interp_df.itertuples()):
    frame_start = time.time()
    img = Image.new("RGBA", RESOLUTION, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # График высоты
    altitude_start = time.time()
    altitude_img = create_altitude_point(static_altitude_img, row, WIDGET_CONFIG['altitude_graph'], min_dist, max_dist, min_alt, max_alt)
    img.paste(altitude_img, WIDGET_CONFIG['altitude_graph']['position'], altitude_img)
    altitude_end = time.time()

    # Мини-карта
    minimap_start = time.time()
    if 'minimap' in WIDGET_CONFIG:
        minimap_config = WIDGET_CONFIG['minimap'].copy()
        map_window = interp_df[
            (interp_df['time_sec'] >= row.time_sec - 30) &
            (interp_df['time_sec'] <= row.time_sec + 30)
            ]
        # minimap = create_minimap(map_window, row._asdict(), minimap_config)
        minimap = create_minimap(map_window, row._asdict(), minimap_config, minimap_background, minimap_bounds)
        img.paste(minimap, minimap_config['position'], minimap)
    minimap_end = time.time()

    # Инициализация глобального словаря для хранения предыдущих значений
    previous_values = {}

    # Текстовые виджеты
    widget_start = time.time()
    for widget_key in text_widgets:
        config = WIDGET_CONFIG[widget_key]
        widget_type = config.get('type', 'static')  # По умолчанию статический

        if widget_type == 'static':
            display_value = config.get('static_text', '')
        else:
            data_field = config['data_field']
            value = getattr(row, data_field)
            if pd.isna(value):
                display_value = ''
            else:
                # Блок сравнения с предыдущим значением
                min_change = config.get('min_change', 0)  # По умолчанию 0%
                if min_change > 0:
                    prev_value = previous_values.get(widget_key, value)  # Инициализация первого значения
                    delta = abs(value - prev_value)
                    max_delta = prev_value * (min_change / 100)  # Процент от предыдущего значения

                    if delta > max_delta:
                        # Ограничиваем изменение до ±min_change%
                        if value > prev_value:
                            value = prev_value + max_delta
                        else:
                            value = prev_value - max_delta

                    # Обновляем предыдущее значение для следующего кадра
                    previous_values[widget_key] = value

                # Форматирование значения
                if config['formatter'] == 'int':
                    formatted_value = f"{int(value)}"
                else:
                    formatted_value = f"{value:{config['formatter']}}"
                display_value = f"{formatted_value}{config.get('postfix', '')}"

        # Остальной код отображения виджета
        render_config = {
            'position': config['position'],
            'font_size': config['font_size']
        }
        create_text_widget(draw, display_value, render_config)
    widget_end = time.time()

    # Отрисовка прогресс-баров
    progress_bar_start = time.time()
    progress_widgets = ['heart_rate_progress', 'power_progress']
    for widget in progress_widgets:
        config = WIDGET_CONFIG.get(widget, {})
        if not config:
            continue
        # Получаем текущее значение
        if widget == 'heart_rate_progress':
            current_value = getattr(row, 'heart_rate', 0)
        elif widget == 'power_progress':
            current_value = getattr(row, 'power', 0)
        else:
            current_value = 0
        # Отрисовываем прогресс-бар
        create_progress_bar(draw, current_value, config, widget)
    progress_bar_end = time.time()

    # Сбор времени
    frame_num = f"{idx + 1:06d}"
    altitude_times.append(altitude_end - altitude_start)
    minimap_times.append(minimap_end - minimap_start)
    widget_times.append(widget_end - widget_start)
    progress_bar_times.append(progress_bar_end - progress_bar_start)
    frame_times.append(time.time() - frame_start)

    img.save(os.path.join(OUTPUT_FOLDER, f"frame_{frame_num}.png"), compress_level=1)

    # Периодический вывод статистики
    current_time = time.time()
    if current_time - last_log_time > 5 or idx == 0 or idx == total_frames - 1:
        frames_done = idx + 1
        avg_frame = np.mean(frame_times[-5:])  # Среднее по последним 5 кадрам
        fps = 1 / avg_frame if avg_frame > 0 else 0
        remaining = total_frames - frames_done
        eta_seconds = remaining * avg_frame
        eta = str(datetime.timedelta(seconds=int(eta_seconds))) if eta_seconds > 0 else "complete"

        stats = (
            f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] "
            f"Сгенерировано: {frames_done}/{total_frames} кадров | "
            f"Среднее время: {avg_frame:.2f}с/кадр | "
            f"FPS: {fps:.1f} | "
            f"ETA: {eta}\n"
            f"  ├─ График: {np.mean(altitude_times) * 1000:.0f}мс\n"
            f"  ├─ Миникарта: {np.mean(minimap_times) * 1000:.0f}мс\n"
            f"  ├─ Виджеты: {np.mean(widget_times) * 1000:.0f}мс\n"
            f"  └─ Прогресс-бары: {np.mean(progress_bar_times) * 1000:.0f}мс"
        )
        print(stats)
        last_log_time = current_time

td = datetime.datetime.now() - time_of_script_start
print(f"\nРендеринг завершен! Общее время: {f"{td.seconds // 3600} часов, {(td.seconds // 60) % 60} минут"}")
