using System;


public static class StringExtender
{
    public static bool NotIn(this string x, params string[] a)
    {
        foreach (string y in a)
        {
            if (x == y)
            {
                return false;
            }
        }

        return true;
    }

    public static bool EqCI(this string x, string y)
    {
        return x.Equals(y, StringComparison.OrdinalIgnoreCase);
    }

    public static bool InCI(this string x, string[] ys)
    {
        foreach( string a in ys )
        {
            if (EqCI(x, a)) return true;
        }

        return false;
    }
}

