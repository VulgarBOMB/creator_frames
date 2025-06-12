# Стилизация
FONT_PATH = r"C:\Windows\Fonts\Square721 Cn BT Bold.ttf"
COLORS = {
    'altitude_line': (255, 255, 255, 255),
    'current_point': (255, 255, 255, 255),
    'text': (255, 255, 255, 255),
    'widget_background': (0, 0, 0, 128),
    'graph_fill': (255, 255, 255, 200)
}

# Позиции и стили виджетов
WIDGET_CONFIG = {
    'altitude_graph': {
        'position': (-150, 808),
        'size': (705, 300),
        'font_size': 42,
        'line_color': COLORS['altitude_line'],
        'line_width': 4,
        'point_size': 6,
        'padding': {
            'left': 141,
            'right': 141,
            'top': 80,
            'bottom': 28.5
        },
        'text_position': 'right',
        'text_offset': {
            'horizontal': 0,
            'vertical': -50
        }
    },
    'minimap': {
        'position': (45, 30),
        'size': (300, 300),
        'map_image': "10.06.2025.png",
        # 55.712163, 37.545618
        # 55.610111, 37.631039
        'map_bounds': {
            'min_lon': 37.545118+0.0002,  # Южная граница
            'max_lon': 37.630539+0.0002,  # Северная граница
            'min_lat': 55.610111+0.0001,  # Западная граница
            'max_lat': 55.712163+0.0001,  # Восточная граница
        },
        'line_color': (255, 255, 255, 128),
        'line_width': 4,
        'point_size': 5,
        'padding': {
            'left': 15,
            'right': 15,
            'top': 15,
            'bottom': 15
        }
    },
    'heart_rate_progress': {
        'position': (50, 475-15),  # Позиция под heart_rate
        'size': (300, 40),  # Размер (ширина, высота)
        'max_value': 220,  # Максимальное значение
        'border_color': (255, 255, 255, 255),
        'fill_color': (255, 255, 255, 255),
        'line_width': 2,
        'font_size': 30,
        'postfix': ' BPM',
    },
    'power_progress': {
        'position': (50, 750-15),  # Позиция под power
        'size': (300, 40),
        'max_value': 500,  # Максимальное значение
        'border_color': (255, 255, 255, 255),
        'fill_color': (255, 255, 255, 255),
        'line_width': 2,
        'font_size': 30,
        'postfix': ' W',
    },
    # Динамические виджеты с данными
    'speed': {
        'position': (150, 555-15),
        'font_size': 105,
        'type': 'dynamic',
        'data_field': 'speed',
        'formatter': '.1f',
        'postfix': ''
    },
    'heart_rate': {
        'position': (7.5, 495),
        'font_size': 105,
        'type': 'dynamic',
        'data_field': 'heart_rate',
        'formatter': 'int',
        'postfix': ''
    },
    'power': {
        'position': (7.5, 675),
        'font_size': 105,
        'type': 'dynamic',
        'data_field': 'power',
        'formatter': 'int',
        'postfix': ''
    },
    'grade': {
        'position': (10, 1020),
        'font_size': 36,
        'type': 'dynamic',
        'data_field': 'grade',
        'formatter': '.1f',
        'postfix': '%',
        'min_change': 5,
    },
    # Статические виджеты (надписи)
    'KMH': {
        'position': (55, 600-15),
        'font_size': 42,
        'type': 'static',
        'static_text': 'KMH'
    },
    'BPM': {
        'position': (7.5, 457.5),
        'font_size': 42,
        'type': 'static',
        'static_text': 'BPM'
    },
    'WATTS': {
        'position': (7.5, 637.5),
        'font_size': 42,
        'type': 'static',
        'static_text': 'WATTS'
    }
}
