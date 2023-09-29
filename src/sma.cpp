#include "sma.h"

SMA::SMA(int nFrames)
{
    m_nFrames = nFrames;
    m_memory = new float[m_nFrames];
    m_filled = false;
    m_cursor = 0;
}

float SMA::newVal(float val)
{
    if (!m_filled)
    {
        m_sms += val;
        m_memory[m_cursor++] = val;

        if (m_cursor == m_nFrames - 1)
        {
            m_filled = true;
        }

        return val;
    }
    else
    {
        m_cursor = (m_cursor + 1) % m_nFrames;
        m_sms = m_sms + val - m_memory[m_cursor];
        m_memory[m_cursor] = val;
        return m_sms / m_nFrames;
    }
}