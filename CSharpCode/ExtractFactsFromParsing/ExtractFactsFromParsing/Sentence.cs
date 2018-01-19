using System;
using System.Collections.Generic;
using System.Xml;


class Sentence
{
    string text;
    List<SToken> tokens;
    public SNode root;
    List<SNode> nodes;

    public Sentence(XmlNode n_sent)
    {
        text = n_sent.SelectSingleNode("text").InnerText;

        // токены
        tokens = new List<SToken>();
        int token_index = 0;
        foreach (XmlNode n_token in n_sent.SelectNodes("tokens/token"))
        {
            SToken t = new SToken(token_index, n_token);
            tokens.Add(t);
            token_index++;
        }

        // дерево зависимостей
        List<int> root_index = new List<int>();
        Dictionary<int, int> child2parent = new Dictionary<int, int>();
        Dictionary<KeyValuePair<int, int>, string> edge_type = new Dictionary<KeyValuePair<int, int>, string>();
        Dictionary<int, List<int>> parent2child = new Dictionary<int, List<int>>();

        foreach (XmlNode n_token in n_sent.SelectNodes("syntax_tree/node"))
        {
            int child_index = int.Parse(n_token["token"].InnerText);

            if (n_token.Attributes["is_root"] != null && n_token.Attributes["is_root"].Value == "true")
                root_index.Add(child_index);
            else
            {
                int parent_index = int.Parse(n_token["parent"].InnerText);
                child2parent.Add(child_index, parent_index);

                edge_type.Add(new KeyValuePair<int, int>(child_index, parent_index), n_token["link_type"].InnerText);

                List<int> child_idx;
                if (!parent2child.TryGetValue(parent_index, out child_idx))
                {
                    child_idx = new List<int>();
                    parent2child.Add(parent_index, child_idx);
                }

                child_idx.Add(child_index);
            }
        }

        nodes = new List<SNode>();
        for (int inode = 0; inode < tokens.Count; ++inode)
        {
            SNode n = new SNode();
            n.index = inode;
            n.word = tokens[inode];
            nodes.Add(n);
        }

        // проставим родителей и детей в каждом узле
        for (int inode = 0; inode < nodes.Count; ++inode)
        {
            SNode node = nodes[inode];

            if (!root_index.Contains(node.index))
            {
                SNode parent_node = nodes[child2parent[node.index]];
                node.parent = parent_node;

                parent_node.children.Add(node);
                parent_node.edge_types.Add(edge_type[new KeyValuePair<int, int>(node.index, parent_node.index)]);
            }
            else
            {
                root = node;
            }
        }
    }

    public string GetText()
    {
        return text;
    }


    public override string ToString()
    {
        return text;
    }

    public SNode GetNodeByIndex(int index)
    {
        return nodes[index];
    }

    public List<SNode> FindNodesByClass(params string[] part_of_speech)
    {
        List<SNode> res = new List<SNode>();
        root.FindNodesByClass(res, part_of_speech);
        return res;
    }

    public bool ContainsWord(Func<string, bool> predicate)
    {
        foreach (var token in tokens)
        {
            if (predicate(token.word))
            {
                return true;
            }
        }

        return false;
    }
}
