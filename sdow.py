import logging
import requests
import os.path
import sqlite3


"""
Helper classes and methods.
"""

WIKIPEDIA_API_URL = 'https://en.wikipedia.org/w/api.php'


def fetch_wikipedia_pages_info(page_ids, database):
  """Fetched page information such as title, URL, and image thumbnail URL for the provided page IDs.

  Args:
    page_title: The page title to validate.

  Returns:
    None

  Raises:
    ValueError: If the provided page title is invalid.
  """
  pages_info = {}

  current_page_ids_index = 0
  while current_page_ids_index < len(page_ids):
    # Query at most 50 pages per request (given WikiMedia API limits)
    end_page_ids_index = min(current_page_ids_index + 50, len(page_ids))

    query_params = {
        'action': 'query',
        'format': 'json',
        'pageids': '|'.join(page_ids[current_page_ids_index:end_page_ids_index]),
        'prop': 'info|pageimages|pageterms',
        'inprop': 'url|displaytitle',
        'piprop': 'thumbnail',
        'pithumbsize': 160,
        'pilimit': 50,
        'wbptterms': 'description',
    }

    current_page_ids_index = end_page_ids_index

    # Identify this client as per Wikipedia API guidelines.
    # https://www.mediawiki.org/wiki/API:Main_page#Identifying_your_client
    headers = {
        'User-Agent': 'Six Degrees of Wikipedia/1.0 (https://www.sixdegreesofwikipedia.com/; wenger.jacob@gmail.com)',
    }

    req = requests.get(WIKIPEDIA_API_URL, params=query_params, headers=headers)

    try:
      pages_result = req.json().get('query', {}).get('pages')
    except ValueError as error:
      # Log and re-raise the exception.
      logging.exception({
          'error': 'Failed to decode MediaWiki API response: "{0}"'.format(error),
          'status_code': req.status_code,
          'response_text': req.text,
      })
      raise error

    for page_id, page in pages_result.iteritems():
      page_id = int(page_id)

      if 'missing' in page:
        # If the page has been deleted since the current Wikipedia database dump, fetch the page
        # title from the SDOW database and create the (albeit broken) URL.
        page_title = database.fetch_page_title(page_id)
        pages_info[page_id] = {
            'title': page_title,
            'url': 'https://en.wikipedia.org/wiki/{0}'.format(page_title)
        }
      else:
        pages_info[page_id] = {
            'title': page['title'],
            'url': page['fullurl']
        }

        thumbnail_url = page.get('thumbnail', {}).get('source')
        if thumbnail_url:
          pages_info[page_id]['thumbnailUrl'] = thumbnail_url

        description = page.get('terms', {}).get('description', [])
        if description:
          pages_info[page_id]['description'] = description[0][0].upper() + description[0][1:]

  return pages_info


def get_sanitized_page_title(page_title):
  """Validates and returns the sanitized version of the provided page title, transforming it into
  the same format used to store pages titles in the database.

  Args:
    page_title: The page title to validate and sanitize.

  Returns:
    The sanitized page title.

  Examples:
    "Notre Dame Fighting Irish"   =>   "Notre_Dame_Fighting_Irish"
    "Farmers' market"             =>   "Farmers\'_market"
    "3.5" Floppy disk"            =>   "3.5\"_Floppy_disk"
    "Nip/Tuck"                    =>   "Nip\\Tuck"

  Raises:
    ValueError: If the provided page title is invalid.
  """
  validate_page_title(page_title)

  return page_title.strip().replace(' ', '_').replace("'", "\\'").replace('"', '\\"')


def get_readable_page_title(sanitized_page_title):
  """Returns the human-readable page title from the sanitized page title.

  Args:
    page_title: The santized page title to make human-readable.

  Returns:
    The human-readable page title.

  Examples:
    "Notre_Dame_Fighting_Irish"   => "Notre Dame Fighting Irish"
    "Farmers\'_market"            => "Farmers' market"
    "3.5\"_Floppy_disk"           => "3.5" Floppy disk"
    "Nip\\Tuck"                   => "Nip/Tuck"
  """
  return sanitized_page_title.strip().replace('_', ' ').replace("\\'", "'").replace('\\"', '"')


def is_str(val):
  """Returns whether or not the provided value is a string type.

  Args:
    val: The value to check.

  Returns:
    bool: Whether or not the provided value is a string type.
  """
  try:
    return isinstance(val, basestring)
  except NameError:
    return isinstance(val, str)


def is_positive_int(val):
  """Returns whether or not the provided value is a positive integer type.

  Args:
    val: The value to check.

  Returns:
    bool: Whether or not the provided value is a positive integer type.
  """
  return val and isinstance(val, int) and val > 0


def validate_page_id(page_id):
  """Validates the provided value is a valid page ID.

  Args:
    page_id: The page ID to validate.

  Returns:
    None

  Raises:
    ValueError: If the provided page ID is invalid.
  """
  if not is_positive_int(page_id):
    raise ValueError((
        'Invalid page ID "{0}" provided. Page ID must be a positive integer.'.format(page_id)
    ))


def validate_page_title(page_title):
  """Validates the provided value is a valid page title.

  Args:
    page_title: The page title to validate.

  Returns:
    None

  Raises:
    ValueError: If the provided page title is invalid.
  """
  if not page_title or not is_str(page_title):
    raise ValueError((
        'Invalid page title "{0}" provided. Page title must be a non-empty string.'.format(
            page_title)
    ))


class InvalidRequest(Exception):
  """Wrapper class for building invalid request error responses."""
  status_code = 400

  def __init__(self, message, status_code=None, payload=None):
    Exception.__init__(self)
    self.message = message
    if status_code is not None:
      self.status_code = status_code
    self.payload = payload

  def to_dict(self):
    result = dict(self.payload or ())
    result['error'] = self.message
    return result




"""
Runs a bi-directional breadth-first search between two Wikipedia articles and returns a list of
the shortest paths between them.
"""


def get_paths(page_ids, visited_dict):
  """Returns a list of paths which go from the provided pages to either the source or target pages.

  Args:
    page_ids: The list of page IDs whose paths to get.

  Returns:
    list(list(int)): A list of lists of page IDs corresponding to paths from the provided page IDs
      to the source or target pages.
  """
  paths = []

  for page_id in page_ids:
    if page_id is None:
      # If the current page ID is None, it is either the source or target page, so return an empty
      # path.
      return [[]]
    else:
      # Otherwise, recursively get the paths for the current page's children and append them to
      # paths.
      current_paths = get_paths(visited_dict[page_id], visited_dict)
      for current_path in current_paths:
        new_path = list(current_path)
        new_path.append(page_id)
        paths.append(new_path)

  return paths


def breadth_first_search(source_page_id, target_page_id, database):
  """Returns a list of shortest paths from the source to target pages by running a bi-directional
  breadth-first search on the graph of Wikipedia pages.

  Args:
    source_page_id: The page at which to start the search.
    target_page_id: The page at which to end the search.
    database: An Database instance which contains methods to query the Wikipedia database.

  Returns:
    list(list(int)): A list of lists of page IDs corresponding to paths from the source page to the
      target page.
  """
  # If the source and target page IDs are identical, return the trivial path.
  if source_page_id == target_page_id:
    return [[source_page_id]]

  paths = []

  # The unvisited dictionaries are a mapping from page ID to a list of that page's parents' IDs.
  # None signifies that the source and target pages have no parent.
  unvisited_forward = {source_page_id: [None]}
  unvisited_backward = {target_page_id: [None]}

  # The visited dictionaries are a mapping from page ID to a list of that page's parents' IDs.
  visited_forward = {}
  visited_backward = {}

  # Set the initial forward and backward depths to 0.
  forward_depth = 0
  backward_depth = 0

  # Continue the breadth first search until a path has been found or either of the unvisited lists
  # are empty.
  while (len(paths) == 0) and ((len(unvisited_forward) != 0) and (len(unvisited_backward) != 0)):
    # Run the next iteration of the breadth first search in whichever direction has the smaller number
    # of links at the next level.
    forward_links_count = database.fetch_outgoing_links_count(unvisited_forward.keys())
    backward_links_count = database.fetch_incoming_links_count(unvisited_backward.keys())

    if forward_links_count < backward_links_count:
      #---  FORWARD BREADTH FIRST SEARCH  ---#
      forward_depth += 1

      # Fetch the pages which can be reached from the currently unvisited forward pages.
      # The replace() bit is some hackery to handle Python printing a trailing ',' when there is
      # only one key.
      outgoing_links = database.fetch_outgoing_links(unvisited_forward.keys())

      # Mark all of the unvisited forward pages as visited.
      for page_id in unvisited_forward:
        visited_forward[page_id] = unvisited_forward[page_id]

      # Clear the unvisited forward dictionary.
      unvisited_forward.clear()

      for source_page_id, target_page_ids in outgoing_links:
        for target_page_id in target_page_ids.split('|'):
          if target_page_id:
            target_page_id = int(target_page_id)
            # If the target page is in neither visited forward nor unvisited forward, add it to
            # unvisited forward.
            if (target_page_id not in visited_forward) and (target_page_id not in unvisited_forward):
              unvisited_forward[target_page_id] = [source_page_id]

            # If the target page is in unvisited forward, add the source page as another one of its
            # parents.
            elif target_page_id in unvisited_forward:
              unvisited_forward[target_page_id].append(source_page_id)

    else:
      #---  BACKWARD BREADTH FIRST SEARCH  ---#
      backward_depth += 1

      # Fetch the pages which can reach the currently unvisited backward pages.
      incoming_links = database.fetch_incoming_links(unvisited_backward.keys())

      # Mark all of the unvisited backward pages as visited.
      for page_id in unvisited_backward:
        visited_backward[page_id] = unvisited_backward[page_id]

      # Clear the unvisited backward dictionary.
      unvisited_backward.clear()

      for target_page_id, source_page_ids in incoming_links:
        for source_page_id in source_page_ids.split('|'):
          if source_page_id:
            source_page_id = int(source_page_id)
            # If the source page is in neither visited backward nor unvisited backward, add it to
            # unvisited backward.
            if (source_page_id not in visited_backward) and (source_page_id not in unvisited_backward):
              unvisited_backward[source_page_id] = [target_page_id]

            # If the source page is in unvisited backward, add the target page as another one of its
            # parents.
            elif source_page_id in unvisited_backward:
              unvisited_backward[source_page_id].append(target_page_id)

    #---  CHECK FOR PATH COMPLETION  ---#
    # The search is complete if any of the pages are in both unvisited backward and unvisited, so
    # find the resulting paths.
    for page_id in unvisited_forward:
      if page_id in unvisited_backward:
        paths_from_source = get_paths(unvisited_forward[page_id], visited_forward)
        paths_from_target = get_paths(unvisited_backward[page_id], visited_backward)

        for path_from_source in paths_from_source:
          for path_from_target in paths_from_target:
            current_path = list(path_from_source) + [page_id] + list(reversed(path_from_target))

            # TODO: This line shouldn't be required, but there are some unexpected duplicates.
            if current_path not in paths:
              paths.append(current_path)

  return paths




"""
Wrapper for reading from and writing to the SDOW database.
"""




class Database(object):
  """Wrapper for connecting to the SDOW database."""

  def __init__(self, sdow_database):
    if not os.path.isfile(sdow_database):
      raise IOError('Specified SQLite file "{0}" does not exist.'.format(sdow_database))


    self.sdow_conn = sqlite3.connect(sdow_database, check_same_thread=False)

    self.sdow_cursor = self.sdow_conn.cursor()

    self.sdow_cursor.arraysize = 1000

  def fetch_page(self, page_title):
    """Returns the ID and title of the non-redirect page corresponding to the provided title,
    handling titles with incorrect capitalization as well as redirects.

    Args:
      page_title: The title of the page to fetch.

    Returns:
      (int, str, bool): A tuple containing the page ID, title, and whether or not a redirect was
      followed.
      OR
      None: If no page exists.

    Raises:
      ValueError: If the provided page title is invalid.
    """
    sanitized_page_title = get_sanitized_page_title(page_title)

    query = 'SELECT * FROM pages WHERE title = ? COLLATE NOCASE;'
    query_bindings = (sanitized_page_title,)
    self.sdow_cursor.execute(query, query_bindings)

    # Because the above query is case-insensitive (due to the COLLATE NOCASE), multiple articles
    # can be matched.
    results = self.sdow_cursor.fetchall()

    if not results:
      raise ValueError(
          'Invalid page title {0} provided. Page title does not exist.'.format(page_title))

    # First, look for a non-redirect page which has exact match with the page title.
    for current_page_id, current_page_title, current_page_is_redirect in results:
      if current_page_title == sanitized_page_title and not current_page_is_redirect:
        return (current_page_id, get_readable_page_title(current_page_title), False)

    # Next, look for a match with a non-redirect page.
    for current_page_id, current_page_title, current_page_is_redirect in results:
      if not current_page_is_redirect:
        return (current_page_id, get_readable_page_title(current_page_title), False)

    # If all the results are redirects, use the page to which the first result redirects.
    query = 'SELECT target_id, title FROM redirects INNER JOIN pages ON pages.id = target_id WHERE source_id = ?;'
    query_bindings = (results[0][0],)
    self.sdow_cursor.execute(query, query_bindings)

    result = self.sdow_cursor.fetchone()

    # TODO: This will no longer be required once the April 2018 database dump occurs since this
    # scenario is prevented by the prune_pages_file.py Python script during the database creation.
    if not result:
      raise ValueError(
          'Invalid page title {0} provided. Page title does not exist.'.format(page_title))

    return (result[0], get_readable_page_title(result[1]), True)

  def fetch_page_title(self, page_id):
    """Returns the page title corresponding to the provided page ID.

    Args:
      page_id: The page ID whose ID to fetch.

    Returns:
      str: The page title corresponding to the provided page ID.

    Raises:
      ValueError: If the provided page ID is invalid or does not exist.
    """
    validate_page_id(page_id)

    query = 'SELECT title FROM pages WHERE id = ?;'
    query_bindings = (page_id,)
    self.sdow_cursor.execute(query, query_bindings)

    page_title = self.sdow_cursor.fetchone()

    if not page_title:
      raise ValueError(
          'Invalid page ID "{0}" provided. Page ID does not exist.'.format(page_id))

    # return page_title[0].encode('utf-8').replace('_', ' ')
    return page_title[0].replace('_', ' ')

  def compute_shortest_paths(self, source_page_id, target_page_id):
    """Returns a list of page IDs indicating the shortest path between the source and target pages.

    Note: the provided page IDs must correspond to non-redirect pages, but that check is not made
    for performance reasons.

    Args:
      source_page_id: The ID corresponding to the page at which to start the search.
      target_page_id: The ID corresponding to the page at which to end the search.

    Returns:
      list(list(int)): A list of integer lists corresponding to the page IDs indicating the shortest path
        between the source and target page IDs.

    Raises:
      ValueError: If either of the provided page IDs are invalid.
    """
    validate_page_id(source_page_id)
    validate_page_id(target_page_id)

    return breadth_first_search(source_page_id, target_page_id, self)

  def fetch_outgoing_links_count(self, page_ids):
    """Returns the sum of outgoing links of the provided page IDs.

    Args:
      page_ids: A list of page IDs whose outgoing links to count.

    Returns:
      int: The count of outgoing links.
    """
    return self.fetch_links_count_helper(page_ids, 'outgoing_links_count')

  def fetch_incoming_links_count(self, page_ids):
    """Returns the sum of incoming links for the provided page IDs.

    Args:
      page_ids: A list of page IDs whose incoming links to count.

    Returns:
      int: The count of incoming links.
    """
    return self.fetch_links_count_helper(page_ids, 'incoming_links_count')

  def fetch_links_count_helper(self, page_ids, incoming_or_outgoing_links_count):
    """Returns the sum of outgoing or incoming links for the provided page IDs.

    Args:
      page_ids: A list of page IDs whose outgoing or incoming links to count.

    Returns:
      int: The count of outgoing or incoming links.
    """
    page_ids = str(tuple(page_ids)).replace(',)', ')')

    # There is no need to escape the query parameters here since they are never user-defined.
    query = 'SELECT SUM({0}) FROM links WHERE id IN {1};'.format(
        incoming_or_outgoing_links_count, page_ids)
    self.sdow_cursor.execute(query)

    return self.sdow_cursor.fetchone()[0]

  def fetch_outgoing_links(self, page_ids):
    """Returns a list of tuples of page IDs representing outgoing links from the list of provided
    page IDs to other pages.

    Args:
      page_ids: A list of page IDs whose outgoing links to fetch.

    Returns:
      list(int, int): A lists of integer tuples representing outgoing links from the list of
        provided page IDs to other pages.
    """
    return self.fetch_links_helper(page_ids, 'outgoing_links')

  def fetch_incoming_links(self, page_ids):
    """Returns a list of tuples of page IDs representing incoming links from the list of provided
    page IDs to other pages.

    Args:
      page_ids: A list of page IDs whose incoming links to fetch.

    Returns:
      list(int, int): A lists of integer tuples representing incoming links from the list of
        provided page IDs to other pages.
    """
    return self.fetch_links_helper(page_ids, 'incoming_links')

  def fetch_links_helper(self, page_ids, outcoming_or_incoming_links):
    """Helper function which handles duplicate logic for fetch_outgoing_links() and
    fetch_incoming_links().

    Args:
      page_ids: A list of page IDs whose links to fetch.
      outcoming_or_incoming_links: String which indicates whether to fetch outgoing ("source_id") or
        incoming ("target_id") links.

    Returns:
      list(int, int): A cursor of a lists of integer tuples representing links from the list of
        provided page IDs to other pages.
    """
    # Convert the page IDs into a string surrounded by parentheses for insertion into the query
    # below. The replace() bit is some hackery to handle Python printing a trailing ',' when there
    # is only one key.
    page_ids = str(tuple(page_ids)).replace(',)', ')')

    # There is no need to escape the query parameters here since they are never user-defined.
    query = 'SELECT id, {0} FROM links WHERE id IN {1};'.format(
        outcoming_or_incoming_links, page_ids)
    self.sdow_cursor.execute(query)

    return self.sdow_cursor