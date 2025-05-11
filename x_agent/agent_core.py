import logging
import os
import platform
import re  # Added for stripping <think> tags
import subprocess

import chromadb
import ollama
from chromadb.config import Settings
from news_fetcher import (
    NewsFetcher,  # Assuming news_fetcher.py is in the same directory
)
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Configuration ---
DB_DIRECTORY = "chroma_db"
COLLECTION_NAME = "tweets"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Consistent model for populating & querying
OLLAMA_MODEL_NAME = "qwen3:1.7b"
NEWS_API_KEY = "39fa1c943e6f40cf98ec4d034099e3a8"
# --- End Configuration ---


class TweetGeneratorAgent:
    def __init__(self):
        self.news_fetcher = NewsFetcher(api_key=NEWS_API_KEY)
        self.embedding_model = None
        self.chroma_collection = None
        self.ollama_available = False

        self._initialize_embedding_model()
        self._initialize_chroma_db()
        self._check_ollama()

    def _initialize_embedding_model(self):
        try:
            logging.info(
                f"Loading sentence transformer model: {EMBEDDING_MODEL_NAME}..."
            )
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logging.info("Sentence transformer model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load sentence transformer model: {e}")
            self.embedding_model = None

    def _initialize_chroma_db(self):
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            db_full_path = os.path.join(script_dir, DB_DIRECTORY)

            client = chromadb.Client(
                Settings(persist_directory=db_full_path, is_persistent=True)
            )
            try:
                self.chroma_collection = client.get_collection(name=COLLECTION_NAME)
                logging.info(
                    f"Successfully connected to ChromaDB collection '{COLLECTION_NAME}' with {self.chroma_collection.count()} items."
                )
            except Exception:
                logging.warning(
                    f"ChromaDB collection '{COLLECTION_NAME}' not found at {db_full_path}. "
                    f"It may need to be created by running populate_db.py."
                )
                self.chroma_collection = None
        except Exception as e:
            logging.error(f"Error initializing ChromaDB client: {e}")
            self.chroma_collection = None

    def _check_ollama(self):
        try:
            ollama.list()
            logging.info(
                f"Ollama server is available. Ready to use model '{OLLAMA_MODEL_NAME}'."
            )
            self.ollama_available = True
        except Exception as e:
            logging.error(
                f"Ollama server not available or model '{OLLAMA_MODEL_NAME}' check failed: {e}"
            )
            logging.warning("Tweet generation will not be possible without Ollama.")
            self.ollama_available = False

    def find_relevant_tweets(self, query_text, n_results=3):
        if not self.chroma_collection or not self.embedding_model:
            logging.warning(
                "ChromaDB collection or embedding model not initialized. Cannot find relevant tweets."
            )
            return []
        try:
            query_embedding_list = self.embedding_model.encode([query_text])
            if query_embedding_list is None or len(query_embedding_list) == 0:
                logging.error("Failed to generate query embedding.")
                return []
            query_embedding = query_embedding_list[0]

            results = self.chroma_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                include=["metadatas"],
            )

            relevant_tweets = []
            metadatas_outer_list = results.get("metadatas")
            if metadatas_outer_list:
                for metadata_list in metadatas_outer_list:
                    if isinstance(metadata_list, list):
                        for item_metadata in metadata_list:
                            if item_metadata and "text" in item_metadata:
                                relevant_tweets.append(item_metadata["text"])

            logging.info(
                f"Found {len(relevant_tweets)} relevant tweets for query: '{query_text}'"
            )
            return relevant_tweets
        except Exception as e:
            logging.error(f"Error querying ChromaDB: {e}")
            return []

    def generate_tweet_draft(
        self, article_title, full_article_content, relevant_past_tweets
    ):
        if not self.ollama_available:
            logging.error("Ollama not available. Cannot generate tweet draft.")
            return None

        max_content_length = 2000
        truncated_content = (
            full_article_content[:max_content_length] + "..."
            if len(full_article_content) > max_content_length
            else full_article_content
        )

        if relevant_past_tweets:
            example_tweets_str = "\n".join(
                [f'- "{tweet}"' for tweet in relevant_past_tweets]
            )
        else:
            example_tweets_str = "No specifically relevant past examples found. Please generate content based on your core style."

        prompt_template = """
You are "Back to Basic," the AI personality behind a popular social media account on X (formerly Twitter).
Your mission is to take current news articles and transform them into highly engaging, super-simplified explanations for a general audience. You're the knowledgeable yet approachable friend who breaks down complex stuff so anyone can get it, often with a relatable hook or a touch of lightheartedness.

**Core "Back to Basic" Style Mandate:**
(Mandate points 1-9 as before...)
1.  **Hook 'Em Early:** Often start with a direct question (e.g., "What's happening with X?", "So, what's the deal with Y? 🤔") or a very brief, attention-grabbing statement about the news.
2.  **Ultra-Simplicity:** Explain the core news as if you're talking to someone smart but completely unfamiliar with the topic. Define key terms or players if necessary (e.g., "The Federal Reserve ('the Fed') is America's central bank.").
3.  **"Why it Matters" / "The Gist":** Crucially, explain *why* this news is important or relevant to the average person. How does it affect them or the world?
4.  **Context is Key (Briefly!):** Provide just enough backstory or context to make the current event understandable (e.g., "For the past year or so, the Fed has been raising interest rates...").
5.  **Anticipate Follow-up Questions:** Think about what a curious reader would ask next and try to address it concisely (e.g., "What investors are really listening for:", "Why are rate cuts often seen as positive?").
6.  **Conversational & Engaging Tone:** Use a friendly, approachable, and conversational voice. A touch of humor or a relatable analogy is great where appropriate, but clarity is paramount. Avoid dry, academic, or overly formal language.
7.  **Visual Appeal & X-Native:**
    *   **Emojis:** Use relevant emojis strategically to add personality and break up text (📈, 🤔, 🌍).
    *   **Hashtags:** ALWAYS include the signature `#BackToBasic`. Consider 1-2 other relevant, common hashtags if space allows.
    *   **Conciseness:** Tweets must be suitable for X.
8.  **Focus:** Distill the *single most important aspect* or the core development from the article for a general audience. Don't try to cover everything.
9.  **No Jargon (or Explain It):** Avoid jargon. If a technical term is absolutely necessary, define it immediately in simple terms.

Here are some relevant examples of past tweets that might help you with the style for this topic:
--- BEGIN RELEVANT EXAMPLES ---
{example_tweets_formatted}
--- END RELEVANT EXAMPLES ---

**Current News Item:**
Headline: {article_title}
Article Snippet/Key Information: {article_content}
(This content should ideally be a concise summary or the most critical paragraphs of the news story you want to explain)

Your post here:
"""
        prompt = prompt_template.format(
            article_title=article_title,
            article_content=truncated_content,
            example_tweets_formatted=example_tweets_str,
        )

        if len(prompt) < 2000:
            logging.info(
                f"Generating tweet with prompt for model {OLLAMA_MODEL_NAME}:\n{prompt}"
            )
        else:
            logging.info(
                f"Generating tweet with prompt for model {OLLAMA_MODEL_NAME} (Prompt is long, content logged at DEBUG level). Example count: {len(relevant_past_tweets or [])}"
            )
            logging.debug(f"Full prompt (length {len(prompt)}): {prompt}")

        try:
            response = ollama.chat(
                model=OLLAMA_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
            )
            generated_text_raw = response["message"]["content"].strip()
            # Remove <think> blocks
            cleaned_text = re.sub(
                r"<think>.*?</think>\s*", "", generated_text_raw, flags=re.DOTALL
            ).strip()
            logging.info(f"Generated tweet draft (cleaned): {cleaned_text}")
            return cleaned_text
        except Exception as e:
            logging.error(f"Error generating tweet with Ollama: {e}")
            return None

    def generate_tweet_from_selected_article(self, article_url, article_title):
        if not article_url or not article_title:
            logging.warning("Article URL and title are required.")
            return None

        logging.info(f"Fetching full content for: {article_title} ({article_url})")
        full_content = self.news_fetcher.get_full_article_content(article_url)

        if not full_content:
            logging.error(
                f"Could not retrieve full content for {article_url}. Cannot generate tweet."
            )
            return None

        logging.info(
            f"Successfully fetched full content (length: {len(full_content)} chars)."
        )

        relevant_tweets = self.find_relevant_tweets(article_title, n_results=3)
        draft = self.generate_tweet_draft(article_title, full_content, relevant_tweets)
        return draft

    def generate_image_prompt(self, article_title, generated_tweet_text):
        if not self.ollama_available:
            logging.error("Ollama not available. Cannot generate image prompt.")
            return None

        prompt_template_image = """
You are an assistant that creates descriptive prompts for an AI image generator (like DALL-E, Midjourney, or Grok's image capabilities).
Based on the following news article title and the tweet generated about it, create a concise and vivid image prompt.
The image should visually represent the core message or theme of the tweet.
Focus on key subjects, actions, atmosphere, and style if applicable. Aim for a prompt that is detailed enough to guide the AI effectively.

Article Title: {article_title}
Generated Tweet: {tweet_text}

Image Prompt for AI:
"""
        prompt = prompt_template_image.format(
            article_title=article_title, tweet_text=generated_tweet_text
        )

        logging.info(
            f"Generating image prompt with prompt for model {OLLAMA_MODEL_NAME}:\n{prompt}"
        )

        try:
            response = ollama.chat(
                model=OLLAMA_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
            )
            generated_image_prompt_raw = response["message"]["content"].strip()
            # Remove <think> blocks
            cleaned_image_prompt = re.sub(
                r"<think>.*?</think>\s*",
                "",
                generated_image_prompt_raw,
                flags=re.DOTALL,
            ).strip()
            logging.info(f"Generated image prompt (cleaned): {cleaned_image_prompt}")
            return cleaned_image_prompt
        except Exception as e:
            logging.error(f"Error generating image prompt with Ollama: {e}")
            return None

    def _display_in_text_editor(self, content, base_filename):
        """Saves content to a file and attempts to open it in the default text editor."""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Construct path to the 'output' directory, which is one level up from script_dir
            output_dir = os.path.join(script_dir, "..", "output")

            # Ensure the output directory exists
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logging.info(f"Created output directory: {output_dir}")

            file_path = os.path.join(output_dir, base_filename)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            logging.info(f"Content saved to: {file_path}")

            current_os = platform.system()
            if current_os == "Windows":
                os.startfile(file_path)
            elif current_os == "Darwin":
                subprocess.call(["open", file_path])
            else:
                subprocess.call(["xdg-open", file_path])
            logging.info(f"Attempted to open {file_path} in the default editor.")
        except Exception as e:
            logging.error(f"Error saving or opening file {base_filename}: {e}")
            print(
                f"\nCould not automatically open {base_filename}. Please find it at: {file_path}"
            )


if __name__ == "__main__":
    logging.info("--- Interactive Tweet Generator Agent ---")
    agent = TweetGeneratorAgent()

    if not agent.news_fetcher or not agent.news_fetcher.newsapi:
        logging.error("News fetcher component is not available. Exiting.")
    elif not agent.embedding_model:
        logging.error("Embedding model is not available. Exiting.")
    elif not agent.chroma_collection:
        logging.warning(
            "ChromaDB collection is not available. Ensure populate_db.py has run for relevant tweet examples."
        )
    elif not agent.ollama_available:
        logging.error(
            f"Ollama server or model {OLLAMA_MODEL_NAME} not available. Exiting."
        )

    if not (
        agent.news_fetcher
        and agent.news_fetcher.newsapi
        and agent.ollama_available
        and agent.embedding_model
    ):
        logging.critical(
            "One or more critical components failed to initialize. Exiting agent."
        )
    else:
        logging.info("Agent initialized successfully. Starting interaction...")

        input_method_choice = ""
        while input_method_choice not in ["1", "2"]:
            input_method_choice = input(
                "\nHow would you like to provide the news story?\n"
                "1. Fetch from News API (select category/article)\n"
                "2. Enter your own news story manually\n"
                "Please enter 1 or 2: "
            ).strip()
            if input_method_choice not in ["1", "2"]:
                print("Invalid choice. Please enter 1 or 2.")

        final_selected_article_title = None
        final_full_article_content = None
        operation_successful = False

        if input_method_choice == "1":
            logging.info("Fetching news from API...")
            categories = [
                "business",
                "entertainment",
                "general",
                "health",
                "science",
                "sports",
                "technology",
            ]
            print("\nAvailable news categories:")
            for i, cat in enumerate(categories):
                print(f"{i + 1}. {cat.capitalize()}")

            selected_category_api = None
            while True:
                try:
                    choice = input(
                        f"Please select a category number (1-{len(categories)}): "
                    )
                    category_index = int(choice) - 1
                    if 0 <= category_index < len(categories):
                        selected_category_api = categories[category_index]
                        break
                    else:
                        print(
                            f"Invalid choice. Please enter a number between 1 and {len(categories)}."
                        )
                except ValueError:
                    print("Invalid input. Please enter a number.")

            logging.info(
                f"Fetching top headlines for category: '{selected_category_api}' (US)..."
            )
            articles_api = agent.news_fetcher.get_top_headlines(
                category=selected_category_api,
                country="us",
                page_size=10,
            )

            if not articles_api:
                logging.warning(
                    f"No articles found for category '{selected_category_api}' or an error occurred."
                )
                operation_successful = False
            else:
                print(f"\nTop headlines for {selected_category_api.capitalize()}:")
                for i, article_api in enumerate(articles_api):
                    print(f"{i + 1}. {article_api['title']}")

                selected_article_url_api = None
                selected_article_title_api = None
                while True:
                    try:
                        article_choice_str = input(
                            f"Select an article number to get the full story (1-{len(articles_api)}), or 0 to skip: "
                        )
                        article_choice = int(article_choice_str)
                        if article_choice == 0:
                            logging.info("User skipped fetching full article via API.")
                            operation_successful = False
                            break
                        article_index = article_choice - 1
                        if 0 <= article_index < len(articles_api):
                            selected_article_url_api = articles_api[article_index][
                                "url"
                            ]
                            selected_article_title_api = articles_api[article_index][
                                "title"
                            ]
                            break
                        else:
                            print(
                                f"Invalid article number. Please enter a number between 1 and {len(articles_api)}, or 0 to skip."
                            )
                    except ValueError:
                        print("Invalid input. Please enter a number.")

                if selected_article_url_api and selected_article_title_api:
                    logging.info(
                        f"User selected article via API: '{selected_article_title_api}'"
                    )
                    retrieved_content = agent.news_fetcher.get_full_article_content(
                        selected_article_url_api
                    )
                    if retrieved_content:
                        final_selected_article_title = selected_article_title_api
                        final_full_article_content = retrieved_content
                        operation_successful = True
                    else:
                        logging.error(
                            f"Could not retrieve full content for {selected_article_url_api} via API."
                        )
                        operation_successful = False
                else:
                    logging.info("No article selected via API for tweet generation.")
                    operation_successful = False

        elif input_method_choice == "2":
            logging.info("Entering news story manually...")
            print("\nPlease provide the details for your news story:")
            title_manual = ""
            while not title_manual:
                title_manual = input("Enter the article title: ").strip()
                if not title_manual:
                    print("Article title cannot be empty.")

            print(
                "Enter the full article content. Press Ctrl+D (Unix) or Ctrl+Z then Enter (Windows) on a new line when done:"
            )
            content_manual_lines = []
            while True:
                try:
                    line = input()
                    content_manual_lines.append(line)
                except EOFError:
                    break
            content_manual = "\n".join(content_manual_lines).strip()
            while not content_manual:
                print("Article content cannot be empty.")
                print(
                    "Re-enter the full article content. Press Ctrl+D (Unix) or Ctrl+Z then Enter (Windows) on a new line when done:"
                )
                content_manual_lines = []
                while True:
                    try:
                        line = input()
                        content_manual_lines.append(line)
                    except EOFError:
                        break
                content_manual = "\n".join(content_manual_lines).strip()

            final_selected_article_title = title_manual
            final_full_article_content = content_manual
            operation_successful = True
            logging.info(
                f"User provided manual article: '{final_selected_article_title}'"
            )

        if (
            operation_successful
            and final_selected_article_title
            and final_full_article_content
        ):
            logging.info(
                f"Proceeding to generate tweet for: '{final_selected_article_title}'"
            )

            relevant_tweets = agent.find_relevant_tweets(
                final_selected_article_title, n_results=3
            )
            draft_tweet = agent.generate_tweet_draft(
                final_selected_article_title,
                final_full_article_content,
                relevant_tweets,
            )

            if draft_tweet:
                logging.info(
                    f"\n--- GENERATED DRAFT TWEET ---\n{draft_tweet}\n------------------------------"
                )
                print(f"\nGenerated Draft Tweet:\n{draft_tweet}")
                agent._display_in_text_editor(draft_tweet, "generated_tweet.txt")

                logging.info(
                    f"Attempting to generate image prompt for: {final_selected_article_title}"
                )
                image_prompt = agent.generate_image_prompt(
                    final_selected_article_title, draft_tweet
                )

                if image_prompt:
                    logging.info(
                        f"\n--- GENERATED IMAGE PROMPT ---\n{image_prompt}\n-----------------------------"
                    )
                    print(f"\nSuggested Image Prompt for Grok:\n{image_prompt}")
                    agent._display_in_text_editor(
                        image_prompt, "generated_image_prompt.txt"
                    )
                else:
                    logging.warning(
                        f"Could not generate an image prompt for: {final_selected_article_title}"
                    )
            else:
                logging.error(
                    f"Failed to generate a draft tweet for the selected article: {final_selected_article_title}"
                )
        elif not operation_successful:
            logging.info(
                "No article content available or operation skipped, so no tweet generated."
            )

    logging.info("--- Interactive Tweet Generator Agent Finished ---")
