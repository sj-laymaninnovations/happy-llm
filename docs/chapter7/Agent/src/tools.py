import datetime
import wikipedia
import requests

# Get the current date and time
def get_current_datetime() -> str:
    """Get the real current date and time.
    :return: A string representation of the current date and time."""
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_datetime

def add(a: float, b: float):
    """Calculate the sum of two floating point numbers.
    :param a: The first floating point number.
    :param b: The second floating point number.
    :return: The sum of two floating point numbers."""
    return str(a + b)

def mul(a: float, b: float):
    """Calculate the product of two floating point numbers.
    :param a: The first floating point number.
    :param b: The second floating point number.
    :return: The product of two floating point numbers."""
    return str(a * b)

def compare(a: float, b: float):
    """Compare the sizes of two floating point numbers.
    :param a: The first floating point number.
    :param b: The second floating point number.
    :return: The string representation of the comparison result."""
    if a > b:
        return f'{a} is greater than {b}'
    elif a < b:
        return f'{b} is greater than {a}'
    else:
        return f'{a} is equal to {b}'

def count_letter_in_string(a: str, b: str):
    """Counts the number of occurrences of a letter in a string.
    :param a: The string to search for.
    :param b: Letters to count.
    :return: The number of times the letter appears in the string."""
    string = a.lower()
    letter = b.lower()
    
    count = string.count(letter)
    return(f"The letter '{letter}' appears {count} times in the string.")

def search_wikipedia(query: str) -> str:
    """Search Wikipedia for the first three page summary of the specified query.
    :param query: The query string to search for.
    :return: A string containing the summary of the first three pages."""
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[: 3]:  # Take the first three page titles
        try:
            # Use the page function of the wikipedia module to get the Wikipedia page object with the specified title.
            wiki_page = wikipedia.page(title=page_title, auto_suggest=False)
            # Get a page summary
            summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
        except (
                wikipedia.exceptions.PageError,
                wikipedia.exceptions.DisambiguationError,
        ):
            pass
    if not summaries:
        return "Wikipedia did not find any suitable results"
    return "\n\n".join(summaries)


def get_current_temperature(latitude: float, longitude: float) -> str:
    """Gets the current temperature for the specified latitude and longitude position.
    :param latitude: latitude coordinates.
    :param longitude: longitude coordinates.
    :return: A string representation of the current temperature."""

    # The URL of the Open Meteo API
    open_meteo_url = "https://api.open-meteo.com/v1/forecast"

    # Request parameters
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m',
        'forecast_days': 1,
    }

    # Send API request
    response = requests.get(open_meteo_url, params=params)

    # Check the response status code
    if response.status_code == 200:
        # Parsing JSON responses
        results = response.json()
    else:
        # Processing the request failed
        raise Exception(f"API Request failed with status code: {response.status_code}")

    # Get the current UTC time
    current_utc_time = datetime.datetime.now(datetime.UTC)

    # Convert a time string to a datetime object
    time_list = [datetime.datetime.fromisoformat(time_str).replace(tzinfo=datetime.timezone.utc) for time_str in
                 results['hourly']['time']]

    # Get the temperature list
    temperature_list = results['hourly']['temperature_2m']

    # Find the index closest to the current time
    closest_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))

    # Get the current temperature
    current_temperature = temperature_list[closest_time_index]

    # Returns the string form of the current temperature
    return f'The temperature is now {current_temperature}°C'
