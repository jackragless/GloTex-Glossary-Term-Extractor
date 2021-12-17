import pickle
database = pickle.load(open("data/sdow_graph.pkl", "rb"))
id2title = pickle.load(open("data/id2title.pkl", "rb"))

def sanitize(page_title):
  return page_title.strip().replace(' ', '_').replace("'", "\\'").replace('"', '\\"')

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


def bfs(source_page_id, target_page_id):

  if database[source_page_id]['redirect']:
    source_page_id = database[source_page_id]['redirect']
  if database[target_page_id]['redirect']:
    target_page_id = database[target_page_id]['redirect']

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
    forward_links_count = [database[k]['outgoing_links_count'] for k in unvisited_forward.keys()]
    backward_links_count = [database[k]['incoming_links_count'] for k in unvisited_backward.keys()]

    if forward_links_count < backward_links_count:
      #---  FORWARD BREADTH FIRST SEARCH  ---#
      forward_depth += 1

      # Fetch the pages which can be reached from the currently unvisited forward pages.
      # The replace() bit is some hackery to handle Python printing a trailing ',' when there is
      # only one key.
      outgoing_links = [(k,list(map(int, database[k]['outgoing_links'].split('|')))) for k in unvisited_forward.keys()]

      # Mark all of the unvisited forward pages as visited.
      for page_id in unvisited_forward:
        visited_forward[page_id] = unvisited_forward[page_id]

      # Clear the unvisited forward dictionary.
      unvisited_forward.clear()

      for source_page_id, target_page_ids in outgoing_links:
        for target_page_id in target_page_ids:
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
      # incoming_links = []
      # for k in unvisited_backward.keys():
      #   for ele in database[k]['incoming_links']:
      #     incoming_links.append((k,ele))
      incoming_links = [(k,list(map(int, database[k]['incoming_links'].split('|')))) for k in unvisited_backward.keys() if database[k]['incoming_links']]

      # print(incoming_links)
      # Mark all of the unvisited backward pages as visited.
      for page_id in unvisited_backward:
        visited_backward[page_id] = unvisited_backward[page_id]

      # Clear the unvisited backward dictionary.
      unvisited_backward.clear()

      for target_page_id, source_page_ids in incoming_links:
        for source_page_id in source_page_ids:
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



def sdowDistance(orig, dest):
  try:
    return len(bfs(id2title[sanitize(orig)],id2title[sanitize(dest)])[0])-1
  except (ValueError,IndexError,TypeError):
    return None