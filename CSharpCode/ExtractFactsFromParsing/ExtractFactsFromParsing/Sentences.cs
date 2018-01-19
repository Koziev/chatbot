using System;
using System.Text;
using System.Xml;

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
