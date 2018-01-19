using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml;
using System.Security.Cryptography;


class SToken
{
    public int index;
    public string word;
    public string lemma;
    public string part_of_speech;
    public List<string> tags;

    public SToken(int _index, XmlNode n_token)
    {
        index = _index;
        word = n_token.SelectSingleNode("word").InnerText;
        lemma = n_token.SelectSingleNode("lemma").InnerText;

        if (n_token.SelectSingleNode("part_of_speech") != null)
            part_of_speech = n_token.SelectSingleNode("part_of_speech").InnerText;

        tags = new List<string>();
        if (n_token.SelectSingleNode("tags") != null)
        {
            tags.AddRange(n_token.SelectSingleNode("tags").InnerText.Split('|'));
        }
    }

    public override string ToString()
    {
        return word;
    }


    public bool ContainsTag(string tag)
    {
        foreach (string t in tags)
            if (t.Equals(tag, StringComparison.CurrentCultureIgnoreCase))
                return true;

        return false;
    }
}

class SNode
{
    public int index;
    public SToken word;
    public SNode parent;
    public List<string> edge_types = new List<string>();
    public List<SNode> children = new List<SNode>();

    public bool children_vector_ready = false;
    public bool parent_vector_ready = false;

    public override string ToString()
    {
        return word.ToString();
    }

    public List<SNode> FindEdge(string edge_type)
    {
        List<SNode> res = new List<SNode>();
        for (int i = 0; i < edge_types.Count; ++i)
            if (edge_types[i].Equals(edge_type, StringComparison.CurrentCultureIgnoreCase))
                res.Add(children[i]);

        return res;
    }

    public bool IsPartOfSpeech(params string[] part_of_speech)
    {
        foreach (var p in part_of_speech)
            if (word.part_of_speech.Equals(p, StringComparison.CurrentCultureIgnoreCase))
                return true;

        return false;
    }

    public bool IsPartOfSpeech(string part_of_speech)
    {
        if (word.part_of_speech.Equals(part_of_speech, StringComparison.CurrentCultureIgnoreCase))
            return true;

        return false;
    }

    public void FindNodesByClass(List<SNode> nodes, params string[] part_of_speech)
    {
        if (IsPartOfSpeech(part_of_speech))
            nodes.Add(this);

        foreach (SNode c in children)
            c.FindNodesByClass(nodes, part_of_speech);

        return;
    }

    public List<SNode> FindNodesByClass(string part_of_speech)
    {
        List<SNode> res = new List<SNode>();

        if (IsPartOfSpeech(part_of_speech))
            res.Add(this);

        foreach (SNode c in children)
            res.AddRange(c.FindNodesByClass(part_of_speech));

        return res;
    }

    public bool EqualsWord(params string[] probes)
    {
        foreach (string probe in probes)
            if (probe.Equals(word.word, StringComparison.CurrentCultureIgnoreCase))
                return true;

        return false;
    }


    public string ConvertTreeString()
    {
        System.Text.StringBuilder b = new StringBuilder();

        List<SNode> all = new List<SNode>();
        all.Add(this);
        all.AddRange(children);

        return string.Join(" ", all.OrderBy(z => z.word.index).Select(z => z.word.word).ToArray());
    }
}

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

class Sentences : IDisposable
{
    System.IO.StreamReader rdr;

    public Sentences(string parsing_path)
    {
        rdr = new System.IO.StreamReader(parsing_path);
    }

    void IDisposable.Dispose()
    {
        rdr.Close();
    }

    Sentence fetched;

    public bool Next()
    {
        fetched = null;

        while (!rdr.EndOfStream)
        {
            string line = rdr.ReadLine();
            if (line == null)
                break;

            if (line.StartsWith("<sentence"))
            {
                System.Text.StringBuilder xmlbuf = new StringBuilder();
                xmlbuf.Append("<?xml version='1.0' encoding='utf-8' ?>");
                xmlbuf.Append("<dataroot>");
                xmlbuf.Append(line);

                while (!rdr.EndOfStream)
                {
                    line = rdr.ReadLine();
                    if (line == null)
                        break;

                    xmlbuf.Append(line);

                    if (line == "</sentence>")
                        break;
                }

                xmlbuf.Append("</dataroot>");

                XmlDocument xml = new XmlDocument();

                xml.LoadXml(xmlbuf.ToString());

                XmlNode n_sent = xml.DocumentElement.SelectSingleNode("sentence");

                fetched = new Sentence(n_sent);

                return true;
            }
        }

        return false;
    }

    public Sentence GetFetched()
    {
        return fetched;
    }
}

class ExtractFactsFromParsing
{
    private static int sample_count = 0;

    private static string last_contituents = "";


    static System.IO.StreamWriter wrt_samples;


    static HashSet<string> stop_words3;

    static ExtractFactsFromParsing()
    {
        string stop_words_str = "сам сама само сами саму самого самих самим самой самими все " +
            "это эта эти этими этот эту этим этой этих этого ее ей им её его их наш ваш твой мой я ты мы вы меня " +
            "тебя вас нас мной тобой вами ими нами мне тебе нам вам им ней нем нём них ними тобою мною него нее неё" +
            "себя себе собой все весь всеми всем всю всей всего всех";
        stop_words3 = new HashSet<string>(stop_words_str.Split());
    }



    static bool IsSuitableNode(SNode n)
    {
        return !IsPunkt(n.word.word) && !char.IsDigit(n.word.word[0]);
    }

    static void CollectChildren(SNode node, List<SNode> children)
    {
        foreach (SNode c in node.children)
        {
            if (IsSuitableNode(c))
            {
                children.Add(c);
            }

            CollectChildren(c, children);
        }
    }

    static bool IsPunkt(string word)
    {
        return word.Length > 0 && char.IsPunctuation(word[0]);
    }

    static void ProcessNode(Sentence sent, SNode node, int max_len)
    {
        // 1. Соберем все подчиненные токены, включая сам node
        List<SNode> nx = new List<SNode>();
        if (IsSuitableNode(node))
        {
            nx.Add(node);
        }

        CollectChildren(node, nx);

        if (nx.Count > 1 && nx.Count <= max_len)
        {
            // Сформируем предложение из этих токенов
            string sample = NormalizeSample(string.Join(" ", nx.OrderBy(z => z.word.index).Select(z => z.word.word).ToArray()));
            if (IsUniqueSample(sample) && sample != last_contituents)
            {
                wrt_samples.WriteLine("{0}", sample);
                sample_count++;
                last_contituents = sample;
            }
        }

        // Для глагольного сказуемого можем отдельно выделить пары с актантами
        if (node.IsPartOfSpeech("ГЛАГОЛ") || node.IsPartOfSpeech("ИНФИНИТИВ") || node.IsPartOfSpeech("БЕЗЛИЧ_ГЛАГОЛ"))
        {
            foreach (SNode c in node.children)
            {
                if (IsSuitableNode(c))
                {
                    nx.Clear();
                    nx.Add(node);
                    nx.Add(c);
                    CollectChildren(c, nx);

                    // Сформируем предложение из этих токенов
                    string sample = NormalizeSample(string.Join(" ", nx.OrderBy(z => z.word.index).Select(z => z.word.word).ToArray()));
                    if (IsUniqueSample(sample) && last_contituents != sample)
                    {
                        wrt_samples.WriteLine("{0}", sample);
                        sample_count++;
                        last_contituents = sample;
                    }
                }
            }
        }

        // Рекурсивно делаем фрагменты из каждого подчиненного узла.
        foreach (var c in node.children)
        {
            ProcessNode(sent, c, max_len);
        }

        return;
    }

    static void WriteSample(string sample)
    {
        if (IsUniqueSample(sample))
        {
            wrt_samples.WriteLine(sample);
        }
        return;
    }

    static void ProcessSentence(Sentence sent, int max_len, string filter)
    {
        if (sent.root.IsPartOfSpeech("ГЛАГОЛ") || sent.root.IsPartOfSpeech("БЕЗЛИЧ_ГЛАГОЛ"))
        {
            if (filter == "3")
            {
                // Пропускаем предложения с подлежащим в 3м лице
                List<SNode> sbj = sent.root.FindEdge("SUBJECT");

                if (sent.ContainsWord(z => stop_words3.Contains(z.ToLower())))
                    return;

                if (sbj.Count == 1 && sbj[0].IsPartOfSpeech("СУЩЕСТВИТЕЛЬНОЕ"))
                {
                    WriteSample(sent.GetText());
                }
            }
            else if (filter == "1s")
            {
                // Есть явное подлежащее в первом лице единственном числе
                List<SNode> sbj = sent.root.FindEdge("SUBJECT");

                if (sbj.Count == 1 && sbj[0].IsPartOfSpeech("МЕСТОИМЕНИЕ") && sbj[0].word.word.Equals("я", StringComparison.OrdinalIgnoreCase))
                {
                    WriteSample(sent.GetText());
                }
            }
            else if (filter == "2s")
            {
                // Есть явное подлежащее во втором лице единственном числе
                List<SNode> sbj = sent.root.FindEdge("SUBJECT");

                if (sbj.Count == 1 && sbj[0].IsPartOfSpeech("МЕСТОИМЕНИЕ") && sbj[0].word.word.Equals("ты", StringComparison.OrdinalIgnoreCase))
                {
                    WriteSample(sent.GetText());
                }
            }
        }

        wrt_samples.Flush();
        return;
    }


    static HashSet<Int64> sample_hashes = new HashSet<Int64>();
    static MD5 md5 = MD5.Create();

    static string NormalizeSample(string str)
    {
        return str.ToLower();
    }

    static bool IsUniqueSample(string str)
    {
        byte[] hash = md5.ComputeHash(System.Text.Encoding.UTF8.GetBytes(str.ToLower()));
        Int64 ihash1 = BitConverter.ToInt64(hash, 0);
        Int64 ihash2 = BitConverter.ToInt64(hash, 8);
        Int64 ihash = ihash1 ^ ihash2;

        if (!sample_hashes.Contains(ihash))
        {
            sample_hashes.Add(ihash);
            return true;
        }
        else
        {
            return false;
        }
    }


    static void Main(string[] args)
    {
        string result_folder = @"f:\tmp";
        List<string> parsed_sentences = new List<string>();

        int MAX_SAMPLE = int.MaxValue;
        int MAX_LEN = int.MaxValue;
        string filter = "3";

        #region Command_Line_Options
        for (int i = 0; i < args.Length; ++i)
        {
            if (args[i] == "-parsing")
            {
                parsed_sentences.Add(args[i + 1]);
                i++;
            }
            else if (args[i] == "-output")
            {
                result_folder = args[i + 1];
                i++;
            }
            else if (args[i] == "-max_samples")
            {
                MAX_SAMPLE = int.Parse(args[i + 1]);
                i++;
            }
            else if (args[i] == "-max_len")
            {
                MAX_LEN = int.Parse(args[i + 1]);
                i++;
            }
            else if (args[i] == "-filter")
            {
                filter = args[i + 1];
                i++;

                if (filter != "1s" && filter != "2s" && filter != "3")
                {
                    throw new ApplicationException($"Unknown filter {filter}");
                }
            }
            else
            {
                throw new ApplicationException(string.Format("Unknown option {0}", args[i]));
            }
        }
        #endregion Command_Line_Options

        wrt_samples = new System.IO.StreamWriter(System.IO.Path.Combine(result_folder, "facts.txt"));


        DateTime start_time = DateTime.Now;

        #region Processing_All_Files
        foreach (string mask in parsed_sentences)
        {
            string[] files = null;
            if (System.IO.Directory.Exists(mask))
            {
                files = System.IO.Directory.GetFiles(mask, "*.parsing.txt");
            }
            else if (mask.IndexOfAny("*?".ToCharArray()) != -1)
            {
                files = System.IO.Directory.GetFiles(System.IO.Path.GetDirectoryName(mask), System.IO.Path.GetFileName(mask));
            }
            else
            {
                files = new string[1] { mask };
            }

            Console.WriteLine("Number of parsing files={0}", files.Length);

            foreach (string file in files)
            {
                if (sample_count >= MAX_SAMPLE)
                    break;

                Console.WriteLine("Processing {0}...", file);

                using (Sentences src = new Sentences(file))
                {
                    while (src.Next() && sample_count < MAX_SAMPLE)
                    {
                        Sentence sent = src.GetFetched();
                        sample_count++;

                        if (sent.root == null)
                            continue;

                        if (sample_count > 0 && (sample_count % 10000) == 0)
                        {
                            Console.Write("{0} samples extracted\r", sample_count);
                        }

                        ProcessSentence(sent, MAX_LEN, filter);
                    }
                }
            }
        }
        #endregion Processing_All_Files


        wrt_samples.Close();

        return;
    }
}
