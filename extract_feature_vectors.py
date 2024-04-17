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

                    for path, node in tree:
                        if isinstance(node, javalang.tree.ClassDeclaration) and node.name == class_name:
                            method_count = len(node.methods)
                            rfc = get_rfc(node)
                            field_count = count_fields(node)
                            implemented_interfaces = get_int(node)
                            bcm, nml, wrd, dcm = get_comments_and_names(node, source_code)
                            max_sz, max_cpx, max_ex, max_ret = get_max_metrics(node)

                            data.append([class_name, method_count, rfc, field_count, implemented_interfaces,
                                         bcm, nml, wrd, dcm, max_sz, max_cpx, max_ex, max_ret])

    df = pd.DataFrame(data, columns=['Class', 'MTH', 'RFC', 'FLD', 'INT',
                                     'BCM', 'NML', 'WRD', 'DCM', 'SZ', 'CPX', 'EX', 'RET'])
    df.to_csv('feature_vector_list.csv', index=False)

    



def get_rfc(node):
    method_invocations = 0
    for path, node in node:
        if isinstance(node, javalang.tree.MethodInvocation):
            method_invocations += 1
    return method_invocations

def count_fields(node):
    field_count = 0
    for path, node in node:
        if isinstance(node, javalang.tree.FieldDeclaration):
            field_count += len(node.declarators)
    return field_count


def get_int(node):
    implemented_interfaces = 0
    for path, node in node:
        if isinstance(node, javalang.tree.ClassDeclaration):
            if node.implements is not None:
                implemented_interfaces += len(node.implements)
    return implemented_interfaces


def get_comments_and_names(node, source_code):
    block_comment_count = 0
    total_method_name_length = 0
    longest_word_length = 0
    word_count = 0
    max_word_count = 0

    # Extract comments using regular expressions
    comments = re.findall(r'/\*.*?\*/|//.*?\n', source_code, re.DOTALL)

    for comment in comments:
        if comment.startswith("/*") and comment.endswith("*/"):
            block_comment_count += 1
            comment_text = re.sub(r'[^\w\s]', '', comment)
            words = comment_text.split()
            for word in words:
                if len(word) > longest_word_length and word.isalnum():
                    longest_word_length = len(word)
            word_count += len(words)
            max_word_count += words.count(longest_word_length)

    # Calculate average method name length
    method_names = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', source_code)
    total_method_name_length = sum(len(name) for name in method_names)

    avg_method_name_length = total_method_name_length / len(method_names) if len(method_names) > 0 else 0

    return block_comment_count, avg_method_name_length, max_word_count, word_count





def get_max_metrics(node):
    max_sz = 0
    max_cpx = 0
    max_ex = 0
    max_ret = 0

    for path, node in node:
        if not isinstance(node, javalang.tree.BlockStatement):
            sz = count_statements(node)
            cpx = count_conditionals_loops(node)
            ex = count_exceptions(node)
            ret = count_return_points(node)

            max_sz = max(max_sz, sz)
            max_cpx = max(max_cpx, cpx)
            max_ex = max(max_ex, ex)
            max_ret = max(max_ret, ret)

    return max_sz, max_cpx, max_ex, max_ret


def count_statements(node):
    statement_count = 0
    for path, node in node:
        if isinstance(node, (javalang.tree.Statement, javalang.tree.BlockStatement)):
            statement_count += 1
    return statement_count

def count_conditionals_loops(node):
    conditional_loop_count = 0
    for path, node in node:
        if isinstance(node, (javalang.tree.IfStatement, javalang.tree.WhileStatement,
                             javalang.tree.DoStatement, javalang.tree.ForStatement,javalang.tree.SwitchStatement)):
            conditional_loop_count += 1
    return conditional_loop_count

def count_exceptions(node):
    exception_count = 0
    for path, node in node:
        if isinstance(node, javalang.tree.MethodDeclaration) and node.throws is not None:
            exception_count += len(node.throws)
    return exception_count



def count_return_points(node):
    return_point_count = 0
    for path, node in node:
        if isinstance(node, (javalang.tree.ReturnStatement, javalang.tree.MethodDeclaration)):
            return_point_count += 1
    return return_point_count

if __name__ == "__main__":
    input_path = "resources/defects4j-checkout-closure-1f/src/com/google/javascript/jscomp"
    parse_source_code(input_path)
