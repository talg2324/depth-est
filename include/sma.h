class SMA
{
    public:
        SMA(int nFrames);
        ~SMA() {delete[] m_memory; };

        bool m_filled;

        float newVal(float val);

    private:
        int m_nFrames;
        int m_cursor;
        float m_sms;
        float *m_memory;
};

