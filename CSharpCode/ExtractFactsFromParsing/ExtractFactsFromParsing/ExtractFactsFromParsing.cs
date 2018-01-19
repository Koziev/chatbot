/* Вспомогательная утилита для подготовки датасета, используемого при обучении
 * чат-бота https://github.com/Koziev/chatbot 
 * 
 * На входе утилита берет результаты синтаксического и морфологического разбора,
 * выполненного утилитой Parser http://solarix.ru/parser.shtml
 * 
 * Результат работы - текстовый файл, в каждой строке которого содержится
 * одно предложение, по возможности содержащее полный предикат с подлежащим-существительным.
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml;
using System.Security.Cryptography;

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
