
// TODO: port to minaudio

#include <sndfile.h>
#include <string.h>

typedef bool (*EmlWaveReadCallback)(int16_t *, int, void *);


#define EML_WAVE_READ_BUFFER_MAX 1024

bool
eml_wave_read(const char *path, int samplerate, int chunksize,
            EmlWaveReadCallback callback, void *user_data)
{

	SNDFILE	*infile;
	SF_INFO sfinfo ;
	int	readcount = 0;

	memset(&sfinfo, 0, sizeof (sfinfo));

    if (chunksize >  EML_WAVE_READ_BUFFER_MAX) {
        return false;
    }
	if (!(infile = sf_open (path, SFM_READ, &sfinfo))) {
		return false;
    }

	if (sfinfo.channels > 1) {
        return false;
    }

	if (sfinfo.samplerate != samplerate) {
        return false;
    }

    static int16_t data[EML_WAVE_READ_BUFFER_MAX];

	while ((readcount = sf_read_short(infile, data, chunksize))) {  
        const bool ok = callback(data, readcount, user_data);
        if (!ok) {
            return false;
        }
	};

	/* Close input and output files. */
	sf_close(infile) ;
    return true;
}
