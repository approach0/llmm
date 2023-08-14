import re
import sys
import time
from timeout import timeout
import requests
from requests.utils import requote_uri
from bs4 import BeautifulSoup
import html
from slimit import ast
from slimit.parser import Parser


def replace_imath(d):
    d = d.replace(r'[imath]', '$')
    d = d.replace(r'[/imath]', '$')
    d = d.replace(r'</em>', '')
    d = d.replace(r'<em class="hl">', '')
    return d


@timeout(seconds=30)
def search_api(keywords=['$x+y=xy$', 'why'],
    topk=3, online=True):

    keywords = list(map(lambda x: 'OR content:' + x, keywords))
    response = requests.get('https://approach0.xyz/search-relay/', params={
        'p': 1,
        'q': ','.join(keywords)
    })

    data = response.json()
    if data['ret_code'] == 0:
        hits = data['hits'][:topk]
        results = []
        for hit in hits:
            url = hit['field_url']
            if online:
                if 'stackexchange' in url:
                    Q, posts = crawl_MSE(url)
                    if Q is None:
                        continue
                    posts = map(
                        lambda x: f'Solution {1+x[0]}:\n\n' + x[1],
                        enumerate(posts)
                    )
                    posts = '\n\n'.join(posts)
                    doc = Q + '\n\n' + posts
                elif 'artofproblemsolving' in url:
                    doc = crawl_AoPS(url)[:512]
                else:
                    raise NotImplemented
            else:
                title = replace_imath(hit['field_title'])
                content = replace_imath(hit['field_content'])
                doc = title + '\n\n' + content

            result = url + '\n\n' + doc
            results.append(result.strip())
        return results
    else:
        return f'Error: {data["ret_str"]}'


def get_curl():
    c = pycurl.Curl()
    c.setopt(c.CONNECTTIMEOUT, 8)
    c.setopt(c.TIMEOUT, 10)
    c.setopt(c.CAINFO, certifi.where())

    # redirect on 3XX error
    c.setopt(c.FOLLOWLOCATION, 1)
    return c


def extract_p_tag_text(soup: BeautifulSoup) -> str:
    txt = ""
    p_tags = soup.find_all("p")
    for p in p_tags:
        if p.text != " ":
            txt += f"{p.text}\n"
    return txt


def crawl_MSE(url):
    try:
        response = requests.get(url)
        post_page = response.content
    except:
        raise
    s = BeautifulSoup(post_page, "html.parser")
    # get title
    question_header = s.find(id="question-header")
    if question_header is None: # page not found anymore
        return None, None
    question_txt = str(question_header.h1.string)

    # get question
    question = s.find(id="question")
    question_txt += '\n\n'
    question_txt += extract_p_tag_text(question)

    # get answers
    answers_array = []
    answers = s.find(id="answers")
    for answer in answers.findAll("div", {"class": "answer"}):
        txt = extract_p_tag_text(answer)
        upvotes = int(answer.attrs['data-score'])
        if 'accepted-answer' in answer.attrs['class']:
            answers_array.append((sys.maxsize, txt))
        elif upvotes > 0:
            answers_array.append((upvotes, txt))
    top_answers = sorted(answers_array, reverse=True)[:2]
    top_answers = [txt for upvotes, txt in top_answers]

    return question_txt, top_answers


def parse_op_name(obj):
    if isinstance(obj, ast.DotAccessor):
        if isinstance(obj.node, ast.Identifier):
            l = f"{obj.node.value}.{obj.identifier.value}"
        else:
            l = f"{parse_op_name(obj.node)}.{obj.identifier.value}"
    elif isinstance(obj, ast.String):
        l = obj.value
        # no using strip here, because it can remove more than 1
        # instance of quotes, which is not desired and can cause issues
        if l.startswith('"') and l.endswith('"'):
            l = l[1:-1]
        l = l.encode().decode("unicode_escape")
    elif hasattr(obj, "value"):
        l = obj.value
    else:
        l = "<UnknownName>"
    return l


def parse_node(node):
    ret = {}
    if hasattr(node, "value") or isinstance(node, ast.DotAccessor):
        return parse_op_name(node)
    if isinstance(node, ast.Object):
        for prop in node.properties:
            l = parse_op_name(prop.left)
            r = parse_node(prop.right)
            ret[l] = r
        return ret
    elif isinstance(node, ast.Array):
        list = []
        for child in node:
            list.append(parse_node(child))
        return list
    elif isinstance(node, ast.FunctionCall):
        return "<FunctionCall>"
    elif isinstance(node, ast.FuncExpr):
        return "<FuncExpr>"
    elif isinstance(node, ast.Program):
        for child in node:
            if isinstance(child, ast.ExprStatement):
                expr = child.expr
                if isinstance(expr, ast.Assign):
                    l = parse_op_name(expr.left)

                    ret[l] = parse_node(expr.right)
    else:
        return "<UnknownRight>"
    return ret


def get_aops_data(page):
    s = BeautifulSoup(page, "html.parser")
    parser = Parser()
    for script in s.findAll("script"):
        if "AoPS.bootstrap_data" in script.string:
            try:
                tree = parser.parse(script.string)
                parsed = parse_node(tree)
                return parsed
            except SyntaxError:
                return None

    return None


def crawl_AoPS(url):
    m = re.search(r'/c[0-9]+h([0-9]+)p([0-9]+)', url)
    topic_id, post_id = int(m.group(1)), int(m.group(2))
    try:
        session = requests.Session()
        response = session.get(url)
        topic_page = response.content
    except:
        raise

    parsed = get_aops_data(topic_page)
    topic_data = parsed["AoPS.bootstrap_data"]["preload_cmty_data"]["topic_data"]
    session_data = parsed["AoPS.session"]

    # get title
    title = html.unescape(topic_data["topic_title"])

    num_posts = int(topic_data["num_posts"])
    posts_data_tmp = topic_data["posts_data"]
    posts_data = []

    # now this is a bit tricky, but if there are more posts
    # than we received, AoPS sens first 15 and last 15 posts,
    # remove all posts that should be shown only from the end
    for post in posts_data_tmp:
        if post["show_from_start"] == "true":
            posts_data.append(post)

    topic_txt = title
    topic_txt += "\n\n"
    for post in posts_data:
        topic_txt += f"{post['post_canonical']}\n\n"

    return topic_txt


def sleepy_search_api(**kargs):
    sleep_time = 1
    while True:
        try:
            time.sleep(sleep_time)
            results = search_api(**kargs)
            break
        except Exception as e:
            print(str(e))
            sleep_time *= 2
    return results


if __name__ == '__main__':
    keywords = [r'$(\sin x)^7 = a \sin 7x + b \sin 5x + c \sin 3x + d \sin x$']
    keywords = ['dilation, centered at the origin, with scale factor -3, takes 4 - 5i to which complex number?']
    print(sleepy_search_api(keywords=keywords))
    #print(crawl_MSE('https://math.stackexchange.com/questions/1134379/find-sin-x7-reduced-in-specific-terms?noredirect=1'))
    #print(crawl_AoPS('https://artofproblemsolving.com/community/c164h1987630p13841342'))
