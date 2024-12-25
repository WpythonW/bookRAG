from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm
from typing import List, Dict, Any
from dataclasses import dataclass
import logging

@dataclass
class Scene:
    description: str
    who: List[str]
    location: str
    key_details: List[str]
    chunk_index: int
    scene_index: int

class ChromaDBManager:
    def __init__(
        self,
        db_path: str = "./chroma_db",
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        chunk_size: int = 500,
        chunk_overlap: int = 0
    ):
        """
        Инициализация менеджера ChromaDB.
        
        Args:
            db_path: Путь к базе данных ChromaDB
            model_name: Название модели для эмбеддингов
            chunk_size: Размер подчанка
            chunk_overlap: Перекрытие между подчанками
        """
        self.client = chromadb.PersistentClient(path=db_path)
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=['\n', '\n\n', '\n\n\n', '\n\n\n\n']
        )
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def create_collections(
        self,
        collection_name: str,
        scenes: List[Scene],
        book_chunks: List[str],
        recreate: bool = True
    ) -> tuple:
        """
        Создание коллекций сцен и чанков.
        
        Args:
            collection_name: Базовое имя для коллекций
            scenes: Список сцен
            book_chunks: Список больших чанков текста
            recreate: Пересоздать коллекции если существуют
            
        Returns:
            tuple: (scenes_collection, chunks_collection)
        """
        scenes_collection_name = f"{collection_name}_scenes"
        chunks_collection_name = f"{collection_name}_chunks"
        
        # Проверяем существование коллекций
        try:
            existing_collections = self.client.list_collections()
            existing_names = [col.name for col in existing_collections]
            
            if scenes_collection_name in existing_names and chunks_collection_name in existing_names and not recreate:
                self.logger.info(f"Коллекции {scenes_collection_name} и {chunks_collection_name} уже существуют")
                return (
                    self.client.get_collection(scenes_collection_name),
                    self.client.get_collection(chunks_collection_name)
                )
        except Exception as e:
            self.logger.error(f"Ошибка при проверке коллекций: {e}")
            
        # Удаляем старые коллекции если нужно
        if recreate:
            for name in [scenes_collection_name, chunks_collection_name]:
                try:
                    self.client.delete_collection(name)
                    self.logger.info(f"Удалена существующая коллекция: {name}")
                except:
                    pass

        # Создаем коллекции
        scenes_collection = self.client.create_collection(
            name=scenes_collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        chunks_collection = self.client.create_collection(
            name=chunks_collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Добавляем сцены
        self._add_scenes(scenes_collection, scenes)
        
        # Добавляем подчанки
        self._add_chunks(chunks_collection, book_chunks)
        
        return scenes_collection, chunks_collection

    def _add_scenes(self, collection, scenes: List[Scene]) -> None:
        """Добавление сцен в коллекцию."""
        self.logger.info("Подготовка сцен...")
        
        # Подготавливаем ВСЕ данные заранее одним списком
        descriptions = ['search_document: ' + scene.description for scene in scenes]
        ids = [f"scene_{scene.chunk_index}_{scene.scene_index}" for scene in scenes]
        metadatas = [{
            "who": ", ".join(scene.who),
            "location": scene.location,
            "key_details": ", ".join(scene.key_details),
            "chunk_index": scene.chunk_index,
            "scene_index": scene.scene_index
        } for scene in scenes]

        self.logger.info(f"Векторизация {len(descriptions)} сцен...")
        
        # Один прямой вызов encode
        embeddings = self.model.encode(
            descriptions,
            batch_size=64,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True,
            device='cpu'
        ).tolist()

        # Один вызов add
        collection.add(
            documents=descriptions,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        self.logger.info(f"Добавлено сцен: {collection.count()}")

    def _add_chunks(self, collection, book_chunks: List[str]) -> None:
        """Добавление подчанков с единым прогресс-баром."""
        self.logger.info("Подготовка подчанков...")
        
        # Подготавливаем все данные
        all_subchunks = []
        all_metadatas = []
        all_ids = []
        global_chunk_counter = 0

        # Разбиваем на подчанки с прогресс-баром
        for parent_idx, large_chunk in enumerate(tqdm(book_chunks, desc="Разбиение на подчанки")):
            subchunks = self.splitter.split_text(large_chunk)
            
            for subchunk_idx, subchunk in enumerate(subchunks):
                all_subchunks.append(subchunk)
                all_metadatas.append({
                    "parent_chunk_index": parent_idx,
                    "subchunk_index": subchunk_idx,
                    "global_chunk_index": global_chunk_counter
                })
                all_ids.append(f"chunk_{parent_idx}_{subchunk_idx}_{global_chunk_counter}")
                global_chunk_counter += 1

        self.logger.info(f"Векторизация {len(all_subchunks)} подчанков...")
        
        embeddings = self.model.encode(
            all_subchunks,
            batch_size=32,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True,
            device='cpu'
        ).tolist()

        collection.add(
            documents=all_subchunks,
            embeddings=embeddings,
            metadatas=all_metadatas,
            ids=all_ids
        )
        
        self.logger.info(f"Добавлено подчанков: {collection.count()}")

    def search(
        self,
        collection_name: str,
        query: str,
        n_scenes: int = 4,
        n_chunks: int = 2,
        n_global_chunks: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Поиск по сценам и связанным чанкам.
        
        Args:
            collection_name: Базовое имя коллекций
            query: Поисковый запрос
            n_scenes: Количество сцен для поиска
            n_chunks: Количество чанков для каждой сцены
            n_global_chunks: Количество глобально найденных чанков
            
        Returns:
            List[Dict]: Результаты поиска
        """
        self.logger.info(f"Поиск по запросу: {query}")
        
        # Получаем коллекции
        scenes_collection = self.client.get_collection(f"{collection_name}_scenes")
        chunks_collection = self.client.get_collection(f"{collection_name}_chunks")
        
        # Векторизуем запрос
        query_vector = self.model.encode(f'search_query: {query}')
        
        # Ищем релевантные сцены
        scene_results = scenes_collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=n_scenes
        )
        
        results = []
        found_chunks = set()  # Для отслеживания уникальных чанков
        
        # Для каждой найденной сцены
        for scene_doc, scene_metadata in zip(
            scene_results['documents'][0],
            scene_results['metadatas'][0]
        ):
            parent_chunk_index = scene_metadata['chunk_index']
            
            # Ищем подчанки
            chunk_results = chunks_collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=n_chunks,
                where={"parent_chunk_index": parent_chunk_index}
            )
            
            # Сохраняем результаты
            scene_result = {
                'scene': {
                    'document': scene_doc,
                    'metadata': scene_metadata
                },
                'chunks': []
            }
            
            # Добавляем уникальные чанки
            for chunk_doc, chunk_metadata in zip(
                chunk_results['documents'][0],
                chunk_results['metadatas'][0]
            ):
                chunk_id = (
                    chunk_metadata['parent_chunk_index'],
                    chunk_metadata['subchunk_index']
                )
                
                if chunk_id not in found_chunks:
                    found_chunks.add(chunk_id)
                    scene_result['chunks'].append({
                        'document': chunk_doc,
                        'metadata': chunk_metadata
                    })
            
            results.append(scene_result)
        
        # Добавляем глобальный поиск по чанкам
        if n_global_chunks > 0:
            global_chunks = chunks_collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=n_global_chunks
            )
            
            # Создаем специальную "глобальную сцену"
            global_scene = {
                'scene': {
                    'document': 'Global Search Results',
                    'metadata': {
                        'who': 'Global context',
                        'location': 'Various locations',
                        'key_details': ['Globally found relevant chunks'],
                        'chunk_index': -1,  # Специальный индекс для глобального поиска
                        'scene_index': -1
                    }
                },
                'chunks': []
            }
            
            # Добавляем глобально найденные чанки
            for chunk_doc, chunk_metadata in zip(
                global_chunks['documents'][0],
                global_chunks['metadatas'][0]
            ):
                chunk_id = (
                    chunk_metadata['parent_chunk_index'],
                    chunk_metadata['subchunk_index']
                )
                
                if chunk_id not in found_chunks:  # Проверяем уникальность
                    found_chunks.add(chunk_id)
                    global_scene['chunks'].append({
                        'document': chunk_doc,
                        'metadata': chunk_metadata
                    })
            
            if global_scene['chunks']:  # Добавляем только если нашли уникальные чанки
                results.append(global_scene)
        
        # Сортируем результаты по глобальному индексу чанков
        for result in results:
            result['chunks'].sort(
                key=lambda x: x['metadata']['global_chunk_index']
            )
        
        # Сортируем сцены, оставляя глобальные результаты в конце
        results.sort(
            key=lambda x: (
                x['scene']['metadata']['chunk_index'] == -1,  # Глобальные результаты в конце
                x['scene']['metadata']['chunk_index']
            )
        )
        
        return results

    def format_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Форматированный вывод результатов поиска.
        
        Args:
            results: Результаты от метода search()
        """
        for scene_idx, result in enumerate(results, 1):
            scene = result['scene']
            print(f"\n=== Сцена {scene_idx} ===")
            print(f"Описание: {scene['document']}")
            print(f"Локация: {scene['metadata']['location']}")
            print(f"Персонажи: {scene['metadata']['who']}")
            
            print("\nСвязанные чанки:")
            for chunk_idx, chunk in enumerate(result['chunks'], 1):
                print(f"\nЧанк {chunk_idx}:")
                # Показываем первые 200 символов текста
                print(f"Текст: {chunk['document'][:200]}...")
                print(
                    f"Индексы: родительский={chunk['metadata']['parent_chunk_index']}, "
                    f"локальный={chunk['metadata']['subchunk_index']}, "
                    f"глобальный={chunk['metadata']['global_chunk_index']}"
                )
    
    def get_complete_story(self, collection_name: str) -> str:
        """
        Извлекает все сцены из коллекции и объединяет их в единый текст в правильном порядке.
        
        Args:
            collection_name: Базовое имя коллекции (без суффикса '_scenes')
            
        Returns:
            str: Объединенный текст всех сцен
        """
        scenes_collection = self.client.get_collection(f"{collection_name}_scenes")
        
        # Получаем все сцены из коллекции
        all_scenes = scenes_collection.get()
        
        # Создаем список кортежей (chunk_index, scene_index, description)
        scene_data = []
        for doc, metadata in zip(all_scenes['documents'], all_scenes['metadatas']):
            scene_data.append((
                metadata['chunk_index'],
                metadata['scene_index'],
                doc,
                metadata['location'],
                metadata['who']
            ))
        
        # Сортируем сначала по chunk_index, затем по scene_index
        scene_data.sort(key=lambda x: (x[0], x[1]))
        
        # Собираем текст с информацией о месте действия и персонажах
        story_parts = []
        for _, _, description, location, characters in scene_data:
            # Убираем префикс 'search_document: ' если он есть
            clean_description = description.replace('search_document: ', '')
            
            scene_text = f"\nLocation: {location}"
            scene_text += f"\nCharacters: {characters}"
            scene_text += f"\n{clean_description}"
            story_parts.append(scene_text)
        
        # Объединяем все части с двойным переносом строки между сценами
        complete_story = "\n\n".join(story_parts)
        
        self.logger.info(f"Собрано {len(scene_data)} сцен в единый текст")
        return complete_story

    def has_collections(self, collection_name: str) -> bool:
        """
        Проверяет существование обеих коллекций (сцен и чанков) по базовому имени.
        
        Args:
            collection_name: Базовое имя коллекций (без суффиксов '_scenes' и '_chunks')
            
        Returns:
            bool: True если обе коллекции существуют, False в противном случае
        """
        try:
            scenes_collection_name = f"{collection_name}_scenes"
            chunks_collection_name = f"{collection_name}_chunks"
            
            existing_collections = self.client.list_collections()
            existing_names = [col.name for col in existing_collections]
            
            has_both = (
                scenes_collection_name in existing_names and 
                chunks_collection_name in existing_names
            )
            
            self.logger.info(
                f"Проверка коллекций: {scenes_collection_name} и {chunks_collection_name}. "
                f"Результат: {'найдены' if has_both else 'не найдены'}"
            )
            
            return has_both
            
        except Exception as e:
            self.logger.error(f"Ошибка при проверке коллекций: {e}")
            return False
        
    def get_base_collection_names(self) -> List[str]:
        """
        Извлекает список уникальных базовых имен коллекций, убирая суффиксы '_scenes' и '_chunks'.
        
        Returns:
            List[str]: Список уникальных базовых имен коллекций
        """
        try:
            # Получаем все коллекции
            collections = self.client.list_collections()
            collection_names = [col.name for col in collections]
            
            # Собираем базовые имена
            base_names = set()
            for name in collection_names:
                if name.endswith('_scenes'):
                    base_name = name[:-7]  # Убираем '_scenes'
                    # Проверяем существование парной коллекции
                    if f"{base_name}_chunks" in collection_names:
                        base_names.add(base_name)
                        
            base_names_list = sorted(list(base_names))
            
            self.logger.info(f"Найдено {len(base_names_list)} базовых коллекций: {base_names_list}")
            return base_names_list
            
        except Exception as e:
            self.logger.error(f"Ошибка при получении списка коллекций: {e}")
            return []
        
    def delete_collections(self, collection_name: str) -> bool:
        """
        Удаляет пару коллекций (сцены и чанки) по базовому имени.
        
        Args:
            collection_name: Базовое имя коллекций (без суффиксов '_scenes' и '_chunks')
            
        Returns:
            bool: True если удаление прошло успешно, False в случае ошибки
        """
        try:
            scenes_collection_name = f"{collection_name}_scenes"
            chunks_collection_name = f"{collection_name}_chunks"
            
            success = True
            error_messages = []
            
            # Пробуем удалить коллекцию сцен
            try:
                self.client.delete_collection(scenes_collection_name)
                self.logger.info(f"Коллекция {scenes_collection_name} успешно удалена")
            except Exception as e:
                success = False
                error_messages.append(f"Ошибка при удалении {scenes_collection_name}: {str(e)}")
                
            # Пробуем удалить коллекцию чанков
            try:
                self.client.delete_collection(chunks_collection_name)
                self.logger.info(f"Коллекция {chunks_collection_name} успешно удалена")
            except Exception as e:
                success = False
                error_messages.append(f"Ошибка при удалении {chunks_collection_name}: {str(e)}")
                
            if error_messages:
                self.logger.error("Ошибки при удалении коллекций:\n" + "\n".join(error_messages))
                
            return success
            
        except Exception as e:
            self.logger.error(f"Общая ошибка при удалении коллекций: {e}")
            return False