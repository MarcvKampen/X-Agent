import logging
import os
import platform
import re  # Added for sanitizing filenames
import subprocess

from newsapi import NewsApiClient
from newspaper import Article  # Added for fetching full article content

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Configuration ---
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
# --- End Configuration ---


class NewsFetcher:
    def __init__(self, api_key=None):  # Made api_key optional
        # Prioritize passed api_key, then environment variable
        effective_api_key = api_key if api_key is not None else NEWS_API_KEY
        if not effective_api_key:
            self.newsapi = None
            logging.error(
                "News API key not provided via argument or NEWS_API_KEY environment variable. NewsFetcher will not work."
            )
            return
        try:
            self.newsapi = NewsApiClient(api_key=effective_api_key)
            logging.info("NewsApiClient initialized successfully.")
        except Exception as e:
            self.newsapi = None
            logging.error(f"Failed to initialize NewsApiClient: {e}")

    def get_top_headlines(
        self,
        query=None,
        sources=None,
        category=None,
        language="en",
        country="us",
        page_size=5,
    ):
        """
        Fetches top headlines from NewsAPI.
        :param query: Keywords or a phrase to search for.
        :param sources: A comma-seperated string of identifiers for the news sources or blogs you want headlines from.
        :param category: The category you want to get headlines for.
                         Possible options: business, entertainment, general, health, science, sports, technology.
        :param language: The 2-letter ISO-639-1 code of the language you want to get headlines for. Default: en.
        :param country: The 2-letter ISO-3166-1 code of the country you want to get headlines for. Default: us.
        :param page_size: The number of results to return per page (request). 20 is the default, 100 is the maximum.
        :return: A list of articles, or None if an error occurs.
        """
        if not self.newsapi:
            logging.error("NewsApiClient not initialized. Cannot fetch headlines.")
            return None

        try:
            logging.info(
                f"Fetching top headlines with params: q='{query}', sources='{sources}', category='{category}', lang='{language}', country='{country}', page_size={page_size}"
            )
            top_headlines = self.newsapi.get_top_headlines(
                q=query,
                sources=sources,
                category=category,
                language=language,
                country=country,
                page_size=page_size,
            )

            if top_headlines.get("status") == "ok":
                articles = top_headlines.get("articles", [])
                logging.info(f"Successfully fetched {len(articles)} articles.")
                # We are interested in title and a brief description/content
                processed_articles = [
                    {
                        "title": article.get("title"),
                        "description": article.get("description"),
                        "url": article.get("url"),
                    }
                    for article in articles
                ]
                return processed_articles
            else:
                logging.error(
                    f"Error fetching headlines from NewsAPI: {top_headlines.get('code')} - {top_headlines.get('message')}"
                )
                return None
        except Exception as e:
            logging.error(f"An exception occurred while fetching headlines: {e}")
            return None

    def get_full_article_content(self, article_url):
        """
        Fetches and parses the full content of an article from its URL.
        :param article_url: The URL of the article.
        :return: The full text content of the article, or None if an error occurs.
        """
        if not article_url:
            logging.error("Article URL is required to fetch full content.")
            return None
        try:
            logging.info(f"Fetching full content for article: {article_url}")
            # Initialize Article object
            article_obj = Article(article_url)
            # Download HTML content
            article_obj.download()
            # Parse the article to extract main content
            article_obj.parse()
            logging.info(
                f"Successfully fetched and parsed article: {article_obj.title}"
            )
            return article_obj.text
        except Exception as e:
            logging.error(
                f"An exception occurred while fetching full article content from {article_url}: {e}"
            )
            return None


if __name__ == "__main__":
    logging.info("--- Interactive News Fetcher ---")
    # Initialize NewsFetcher. It will try to use NEWS_API_KEY from env if no arg is passed.
    fetcher = NewsFetcher()

    if not NEWS_API_KEY:
        logging.error(
            "NEWS_API_KEY environment variable not set. Please set it to your News API key to run this script."
        )
    elif not fetcher.newsapi:  # Check if initialization failed for other reasons
        logging.error(
            "News fetcher could not be initialized. This might be due to an invalid API key or network issues. Exiting."
        )
    else:
        # Define a list of categories for the user to choose from
        categories = [
            "business",
            "entertainment",
            "general",
            "health",
            "science",
            "sports",
            "technology",
        ]

        # Main interaction loop allowing restart from category selection
        while True:
            print("\nAvailable news categories:")
            for i, cat in enumerate(categories):
                print(f"{i + 1}. {cat.capitalize()}")

            selected_category = None
            # Category selection loop
            while True:
                try:
                    choice = input(
                        f"Please select a category number (1-{len(categories)}), or 0 to exit: "
                    )
                    if not choice.strip():  # Handle empty input
                        print("Invalid input. Please enter a number.")
                        continue
                    category_choice_num = int(choice)
                    if category_choice_num == 0:
                        logging.info("User chose to exit.")
                        # This will break the inner loop, then we need to break outer.
                        selected_category = "EXIT"  # Signal to exit outer loop
                        break
                    category_index = category_choice_num - 1
                    if 0 <= category_index < len(categories):
                        selected_category = categories[category_index]
                        break
                    else:
                        print(
                            f"Invalid choice. Please enter a number between 1 and {len(categories)}, or 0 to exit."
                        )
                except ValueError:
                    print("Invalid input. Please enter a number.")

            if selected_category == "EXIT":
                break  # Exit main interaction loop

            logging.info(
                f"Fetching top headlines for category: '{selected_category}' (US)..."
            )
            articles = fetcher.get_top_headlines(
                category=selected_category, country="us", page_size=10
            )

            if not articles:
                logging.warning(
                    f"No articles found for category '{selected_category}' or an error occurred."
                )
                print(
                    f"\nNo articles found for category '{selected_category}'. Returning to category selection.\n"
                )
                continue  # Restart main interaction loop (back to category selection)

            print(f"\nTop headlines for {selected_category.capitalize()}:")
            for i, article_item in enumerate(articles):
                print(f"{i + 1}. {article_item['title']}")

            selected_article_url = None
            selected_article_title = None
            article_choice = -1  # Initialize to a value that's not 0

            # Article selection loop
            while True:
                try:
                    article_choice_str = input(
                        f"Select an article number to get the full story (1-{len(articles)}), or 0 to return to categories: "
                    )
                    if not article_choice_str.strip():  # Handle empty input
                        print("Invalid input. Please enter a number.")
                        continue
                    article_choice = int(article_choice_str)
                    if article_choice == 0:
                        logging.info("User chose to return to category selection.")
                        break  # Exit article selection, will continue outer loop
                    article_index = article_choice - 1
                    if 0 <= article_index < len(articles):
                        selected_article_url = articles[article_index]["url"]
                        selected_article_title = articles[article_index]["title"]
                        break  # Valid article selected
                    else:
                        print(
                            f"Invalid article number. Please enter a number between 1 and {len(articles)}, or 0 to return to categories."
                        )
                except ValueError:
                    print("Invalid input. Please enter a number.")

            if article_choice == 0:  # User chose 0 in article selection
                print("Returning to category selection...\n")
                continue  # Restart main interaction loop

            # If we reach here, a specific article (not 0) was chosen.
            if selected_article_url:
                logging.info(
                    f"Fetching full story for: {selected_article_title} ({selected_article_url})"
                )
                full_content = fetcher.get_full_article_content(selected_article_url)

                if full_content:
                    logging.info("Full article content fetched successfully.")
                    logging.info(f"Content length: {len(full_content)} characters.")
                    sanitized_title = re.sub(
                        r'[\\/*?:"<>|]', "", selected_article_title or ""
                    )
                    filename = f"{sanitized_title[:100]}.txt"
                    try:
                        with open(filename, "w", encoding="utf-8") as f:
                            f.write(full_content)
                        logging.info(f"Full article content saved to: {filename}")
                        print(f"\n--- Full article content saved to: {filename} ---")
                        try:
                            current_os = platform.system()
                            if current_os == "Windows":
                                os.startfile(filename)
                            elif current_os == "Darwin":
                                subprocess.call(["open", filename])
                            else:
                                subprocess.call(["xdg-open", filename])
                            logging.info(
                                f"Attempted to open {filename} in the default editor."
                            )
                        except Exception as e_open:
                            logging.error(
                                f"Could not automatically open file {filename}: {e_open}"
                            )
                            print(
                                f"Could not automatically open file. Please find it at: {filename}"
                            )

                        logging.info(
                            f"Successfully processed article: {selected_article_title}. Exiting."
                        )
                        break  # Successful processing, exit main interaction loop
                    except Exception as e_save:
                        logging.error(
                            f"Error saving article to file {filename}: {e_save}"
                        )
                        print(f"\nError saving article to file: {filename}")
                        print("\n--- Full Article Content (fallback print) ---")
                        print(full_content)
                        print("--- End of Article (fallback print) ---")
                        print("Returning to category selection due to save error...\n")
                        continue  # Error during save, restart main loop
                else:
                    # Failed to retrieve full_content
                    logging.error(
                        f"Could not retrieve full content for '{selected_article_title}' ({selected_article_url}). It might be behind a paywall or inaccessible."
                    )
                    print(
                        f"\nFailed to retrieve full content for '{selected_article_title}'."
                    )
                    print(
                        "This can happen due to paywalls or other access restrictions."
                    )
                    print("Returning to category selection...\n")
                    continue  # Restart main interaction loop
            # else: # This case should not be hit if article_choice was > 0
            #    logging.warning("Unexpected state: article chosen but URL not set. Restarting.")
            #    continue

    logging.info("--- Interactive News Fetcher Finished ---")
