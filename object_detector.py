import numpy as np
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

def detect_objects_in_image(image_path, top_k=5):
    """
    Распознает объекты на изображении и возвращает список предметов.
    Автоматически переводит названия на указанный язык (по умолчанию русский).

    Args:
        image_path: путь к файлу изображения (jpg, png)
        top_k: количество лучших предположений для вывода (по умолчанию 5)
        target_lang: код целевого языка для перевода (например, 'ru', 'en', 'es')

    Returns:
        Список словарей с ключами: 'object_ru', 'object_en', 'confidence'
        или None в случае ошибки загрузки/обработки изображения
    """
    try:
        # 123123123
        # 1. Ленивая загрузка модели детектора
        if not hasattr(detect_objects_in_image, 'model'):
            print("Загрузка модели MobileNetV2 для детекции...")
            detect_objects_in_image.model = MobileNetV2(weights='imagenet')
            print("Модель детекции загружена!")
        model = detect_objects_in_image.model

        # 2. Загрузка и предобработка изображения
        image = Image.open(image_path).convert('RGB')
        image_resized = image.resize((224, 224))
        image_array = np.array(image_resized)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)

        # 3. Детекция объектов
        predictions = model.predict(image_array, verbose=0)
        decoded_predictions = decode_predictions(predictions, top=top_k)[0]

        # 4. Подготовка списка английских названий для перевода
        english_labels = [label for _, label, _ in decoded_predictions]

        # 5. Попытка перевода с помощью googletrans
        # 5. Попытка перевода с помощью библиотеки translate
        try:
            from translate import Translator
            # Создаем объект переводчика (английский -> русский)
            translator = Translator(from_lang="en", to_lang="ru")
            # Переводим каждую метку по отдельности
            translated_labels = []
            for label in english_labels:
                translated = translator.translate(label.replace('_', ' '))
                translated_labels.append(translated)
            print("Перевод выполнен успешно.")
        except Exception as e:
            # Если перевод не удался, используем оригинальные названия
            print(f"Не удалось выполнить перевод ({e}). Используются оригинальные названия.")
            translated_labels = english_labels

        # 6. Формирование результата
        results = []
        for i, (_, label_en, confidence) in enumerate(decoded_predictions):
            confidence_percent = round(confidence * 100, 1)
            label_translated = translated_labels[i]

            results.append({
                'object': label_translated,      # Переведенное название
                'object_en': label_en,          # Оригинальное английское название (на всякий случай)
                'confidence': confidence_percent # Уверенность модели
            })

        return results

    except FileNotFoundError:
        print(f"Ошибка: Файл '{image_path}' не найден.")
        return None
    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")
        return None

# Функция для пакетной обработки (если нужно проанализировать несколько изображений)
def detect_objects_in_multiple_images(image_paths, top_k=3):
    """
    Обрабатывает несколько изображений и возвращает объединенный список объектов.
    """
    all_objects = []
    for img_path in image_paths:
        print(f"\nАнализ: {img_path}")
        objects = detect_objects_in_image(img_path, top_k=top_k)
        if objects:
            all_objects.extend(objects)
    return all_objects


# Пример использования
if __name__ == "__main__":
    # 1. Убедитесь, что установлены все зависимости:
    # pip install pillow tensorflow numpy googletrans==4.0.0-rc1

    # 2. Протестируйте на вашем изображении
    test_image = "img_1.png"  # Замените на путь к вашему изображению

    print("=" * 50)
    print("ДЕТЕКТОР ОБЪЕКТОВ С АВТОПЕРЕВОДОМ")
    print("=" * 50)

    detected_items = detect_objects_in_image(test_image, top_k=10)

    if detected_items:
        print(f"\nНайдено предметов: {len(detected_items)}")
        for i, item in enumerate(detected_items, 1):
            print(f"{i}. {item['object']} (англ.: {item['object_en']}) - {item['confidence']}%")
    else:
        print("Не удалось распознать предметы на изображении.")