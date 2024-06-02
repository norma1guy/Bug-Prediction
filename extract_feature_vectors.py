import os
import javalang
import re
import pandas as pd

def parse_source_code(input_path):
    data = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith('.java'):
                with open(os.path.join(root, file), 'r') as f:
                    source_code = f.read()
                    tree = javalang.parse.parse(source_code)
                    class_name = os.path.splitext(file)[0]
                    
                    for path, node in tree.filter(javalang.tree.ClassDeclaration):
                        if node.name == class_name:
                            method_count = len(node.methods)
                            rfc = get_rfc(node)
                            field_count = count_fields(node)
                            implemented_interfaces = len(node.implements) if node.implements else 0
                            bcm, nml, wrd, dcm = get_comments_and_names(node, source_code)
                            max_sz, max_cpx, max_ex, max_ret = get_max_metrics(node)

                            data.append([class_name, method_count, field_count, rfc, implemented_interfaces,
                                         max_sz, max_cpx, max_ex, max_ret, bcm, nml, wrd, dcm])

    df = pd.DataFrame(data, columns=['Class', 'MTH', 'FLD', 'RFC', 'INT',
                                     'SZ', 'CPX', 'EX', 'RET', 'BCM', 'NML', 'WRD', 'DCM'])
    df.to_csv('feature_vector_list.csv', index=False)

def get_rfc(node):
    method_invocations = 0
    public_method_count = 0
    for _, n in node:
        if isinstance(n, javalang.tree.MethodInvocation):
            method_invocations += 1
        if isinstance(n, javalang.tree.MethodDeclaration) and 'public' in n.modifiers:
            public_method_count += 1
    return public_method_count + method_invocations

def count_fields(node):
    field_count = 0
    for _, n in node:
        if isinstance(n, javalang.tree.FieldDeclaration):
            field_count += len(n.declarators)
    return field_count

def get_comments_and_names(node, source_code):
    block_comment_count = 0
    total_method_name_length = 0
    longest_word_length = 0
    word_count = 0
    statement_count = 0

    comments = re.findall(r'/\*.*?\*/|//.*?\n', source_code, re.DOTALL)

    for comment in comments:
        if comment.startswith("/*") and comment.endswith("*/"):
            block_comment_count += 1
            comment_text = re.sub(r'[^\w\s]', '', comment)
            words = comment_text.split()
            word_count += len(words)
            longest_word_length = max(longest_word_length, max((len(word) for word in words), default=0))

    method_names = [method.name for _, method in node.filter(javalang.tree.MethodDeclaration)]
    total_method_name_length = sum(len(name) for name in method_names)
    avg_method_name_length = total_method_name_length / len(method_names) if method_names else 0

    for _, stmt in node.filter(javalang.tree.Statement):
        statement_count += 1

    dcm = word_count / statement_count if statement_count else 0

    return block_comment_count, avg_method_name_length, longest_word_length, dcm

def get_max_metrics(node):
    max_sz = 0
    max_cpx = 0
    max_ex = 0
    max_ret = 0

    for _, n in node.filter(javalang.tree.MethodDeclaration):
        sz = count_statements(n)
        cpx = count_conditionals_loops(n)
        ex = len(n.throws) if n.throws else 0
        ret = count_return_points(n)

        max_sz = max(max_sz, sz)
        max_cpx = max(max_cpx, cpx)
        max_ex = max(max_ex, ex)
        max_ret = max(max_ret, ret)

    return max_sz, max_cpx, max_ex, max_ret

def count_statements(node):
    return sum(1 for _ in node.filter(javalang.tree.Statement))

def count_conditionals_loops(node):
    return sum(1 for _ in node.filter((javalang.tree.IfStatement, javalang.tree.WhileStatement,
                                       javalang.tree.DoStatement, javalang.tree.ForStatement,
                                       javalang.tree.SwitchStatement)))

def count_return_points(node):
    return sum(1 for _ in node.filter(javalang.tree.ReturnStatement))

if __name__ == "__main__":
    input_path = "resources/defects4j-checkout-closure-1f/src/com/google/javascript/jscomp"
    parse_source_code(input_path)
