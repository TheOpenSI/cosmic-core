### Core modules ###
from ollama import Client
from threading import Thread
from time import sleep


### Type hints ###


### Internal modules ###
from ....utils.log_tool import set_color



class OllamaPullManager:
    def __init__(
        self,
        model_name:         str,
        mode:               str             = "auto",
        interventions:      list            = [80, 90, 95],
        min_speed_kbps:     float           = 200.0,
        max_retries:        int             = 5,
        fall_back_interval: int             = 60,
        ollama_client:      Client | None   = None
    ) -> None:
        """
        OllamaPullManager to mange fail-safe model pulling

        Args:
            model_name          (str):              Name of the model to pull
            mode                (str, optional):    auto | speed | stochastic. Defaults to "auto".
            interventions       (list, optional):   List of percentages to intervene at. Defaults to [85].
            min_speed_kbps      (float, optional):  Minimum internet speed in kbps. Defaults to 200.0.
            max_retries         (int, optional):    Maximum retries for pulling the model. Defaults to 5.
            fall_back_interval  (int, optional):    Fallback interval in seconds. Defaults to 60.
        """
        self.model_name         = model_name
        self.mode               = mode  # "auto", "speed", "stochastic"
        self.interventions      = interventions
        self.min_speed_kbps     = min_speed_kbps
        self.max_retries        = max_retries
        self.fall_back_interval = fall_back_interval
        self._download_error    = None

        if ollama_client is None:
            self.ollama_client = Client()

        else:
            self.ollama_client = ollama_client

        # Limit interventions to 5 (for Stochastic mode)
        if len(self.interventions) > 5:
            self.interventions = self.interventions[:5]

        # Thread management
        self._is_pulling        = False
        self._should_stop       = False
        self._download_thread   = None

        # Progress tracking
        self._current_percentage    = 0
        self._last_completed        = 0
        self._download_completed    = False

        # Intervention tracking
        self._intervention_count        = 0
        self._completed_interventions   = []

        return None


    def _reset(self) -> None:
        """
        Reset all after a pull.
        """
        self._is_pulling            = False
        self._should_stop           = False
        self._download_thread       = None
        self._current_percentage    = 0
        self._last_completed        = 0
        self._download_completed    = False
        self._intervention_count    = 0

        self._completed_interventions.clear()

        return None


    def _get_available_models(self) -> list[str]:
        """
        Get the model names from the ollama.list() output.

        Returns:
            list[str]: List of available model names.
        """
        available_models = self.ollama_client.list()

        return [
            model.get("model")
            for model in available_models.get("models", [])
        ]


    def _is_model_available(self) -> bool:
        """
        Check if the model is avaiable on the server.
        """
        available_models: list[str] = self._get_available_models()

        return (
            True
            if   (self.model_name in available_models)
            else (False)
        )


    def pull_model(self) -> None:
        """
        Pull the model with the specified mode
        """
        flag: bool = self._is_model_available()

        if flag:
            print(
                set_color(
                    status="info",
                    information=f"Model {self.model_name} is available on the server."
                )
            )
            return None

        print(f"Pulling model '{self.model_name}' in {self.mode} mode...")

        attempt: int = 0
        while attempt < self.max_retries:
            try:
                if self.mode == "stochastic":
                    success = self._pull_with_stochastic()

                elif self.mode == "speed":
                    success = self._pull_with_speed_monitoring()

                else:  # auto mode
                    success = self._pull_with_auto()

                if success:
                    self._reset()
                    print(f"\nModel {self.model_name} pulled successfully!")
                    return None

                if self._download_error is not None:
                    # incorrect model name
                    if "file does not exist" in str(self._download_error).lower():
                        print(f"Model '{self.model_name}' does not exist on the Ollama registry.")

                    # storage issues
                    elif "no space left on device" in str(self._download_error).lower():
                        print("Storage issue detected: No space left on device.")

                    # generic error
                    else:
                        print(f"Download error: {self._download_error}")

                    self._reset()
                    return None

            except Exception as e:
                print(f"\nError in attempt {attempt + 1}: {e}")

            attempt += 1
            if attempt < self.max_retries:
                print(f"Retrying... (Attempt {attempt + 1}/{self.max_retries})")
                sleep(2)

        self._reset()
        print(f"\nFailed to pull model after {self.max_retries} attempts.")


    def _pull_with_stochastic(self) -> bool:
        """
        Pull with stochastic interventions
        """
        for intervention_idx, target_percentage in enumerate(self.interventions):
            if self._intervention_count >= 5:  # Max 5 interventions
                break

            print(
                "{0:s}{1:s}".format(
                    f"Starting download",
                    f"(Intervention {intervention_idx + 1}/{len(self.interventions)} at {target_percentage}%)..."
                )
            )

            self._start_download()
            sleep(5) # Allow some time to start the thread and get initial progress

            if self._download_error is not None:
                print(f"\nDownload error: {self._download_error}")
                return False

            # Monitor progress until target percentage
            while self._is_pulling:
                if self._current_percentage >= target_percentage:
                    self._intervention_count += 1
                    self._completed_interventions.append({
                        'intervention': intervention_idx + 1,
                        'target_percentage': target_percentage,
                        'actual_percentage': self._current_percentage,
                        'status': 'completed'
                    })

                    print(f"\nIntervention {intervention_idx + 1}/5 at {self._current_percentage:.1f}% - "
                          "Restarting...")
                    self._stop_download()
                    sleep(2)  # Brief pause before restart
                    break

                sleep(1)

        if self._download_completed:
            return True
        else:
            return self._fall_back_intervention()


    def _pull_with_speed_monitoring(self) -> bool:
        """Pull with speed monitoring - placeholder"""
        print("Speed monitoring mode - placeholder implementation")

        # TODO: speed monitoring logic
        return self._pull_basic()


    def _pull_with_auto(self) -> bool:
        """Pull with auto detection - placeholder"""
        print("Auto mode - placeholder implementation")

        # TODO: auto detection logic
        return self._pull_basic()


    def _pull_basic(self) -> bool:
        """Basic pull without interventions"""
        self._start_download()

        while self._is_pulling:
            sleep(1)

        return self._download_completed


    def _start_download(self) -> None:
        """
        Start download in a separate thread
        """
        self._should_stop           = False
        self._download_completed    = False
        self._download_thread       = Thread(
            target=self._download_worker,
            daemon=True
        )

        self._download_thread.start()

        return None


    def _stop_download(self) -> None:
        """
        Stop the download thread
        """
        self._should_stop = True
        if self._download_thread and self._download_thread.is_alive():
            self._download_thread.join(timeout=5)

        return None


    def _download_worker(self) -> None:
        """
        Worker function that runs in the thread
        """
        try:
            self._is_pulling = True
            stream = self.ollama_client.pull(self.model_name, stream=True)

            for chunk in stream:
                if self._should_stop:
                    break

                if chunk.get("status"):
                    status = chunk["status"]

                    if "pulling" in status.lower():
                        if 'completed' in chunk and 'total' in chunk:
                            completed = chunk['completed']
                            total = chunk['total']
                            self._current_percentage = (completed / total) * 100
                            self._last_completed = completed

                            # Convert to MB for display
                            completed_mb = completed / (1024 * 1024)
                            total_mb = total / (1024 * 1024)

                            print(f"\rProgress: {self._current_percentage:.1f}% "
                                  f"({completed_mb:.1f}MB/{total_mb:.1f}MB)",
                                  end="", flush=True)

                    elif 'success' in status.lower():
                        self._download_completed = True
                        print(f"\nModel {self.model_name} pulled successfully!")
                        break

                    else:
                        print(f"\r{status}", end="", flush=True)

            return None

        except Exception as e:
            self._download_error = str(e)

        finally:
            self._is_pulling = False


    def _fall_back_intervention(self) -> bool:
        """
        Fallback method to pull the model with set interventions
        """
        print(f"Starting final download attempt with {self.fall_back_interval}s interventions...")
        max_attempts = 10

        for attempt in range(max_attempts):
            print(f"{self.fall_back_interval}s interval attempt {attempt + 1}/{max_attempts}")
            self._start_download()

            for _ in range(5):
                sleep(self.fall_back_interval/5)
                # Check if completed
                if self._download_completed:
                    print("Download completed!")
                    return True

            print(f"{self.fall_back_interval}s interval reached, restarting...")
            self._stop_download()
            sleep(2)

        return False


    def get_intervention_log(self) -> list[str]:
        """
        Get log of completed interventions
        """
        return self._completed_interventions